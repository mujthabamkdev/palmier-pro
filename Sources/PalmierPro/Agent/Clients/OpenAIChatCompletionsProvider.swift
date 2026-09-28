import Foundation

/// `/chat/completions` request body. Deliberately sends only what every OpenAI-compatible
/// server accepts: no `max_tokens` (a fixed budget overflows small local contexts), no
/// reasoning knobs, no store/response options.
enum OpenAIChatCompletionsRequestBody {
    static func build(
        modelID: String,
        system: String,
        tools: [AgentToolSchema],
        messages: [AgentRequestMessage]
    ) -> [String: Any] {
        var body: [String: Any] = [
            "model": modelID,
            "stream": true,
            "messages": chatMessages(system: system, messages: messages),
        ]
        if !tools.isEmpty {
            body["tools"] = tools.map {
                [
                    "type": "function",
                    "function": [
                        "name": $0.name,
                        "description": $0.description,
                        "parameters": $0.inputSchema,
                    ],
                ]
            }
        }
        return body
    }

    private static func chatMessages(
        system: String,
        messages: [AgentRequestMessage]
    ) -> [[String: Any]] {
        var out: [[String: Any]] = []
        if !system.isEmpty {
            out.append(["role": "system", "content": system])
        }
        for message in messages {
            var text = ""
            var parts: [[String: Any]] = []
            var toolCalls: [[String: Any]] = []
            let role = message.role.rawValue

            for block in message.content {
                switch block {
                case .image(let base64, let mediaType):
                    parts.append([
                        "type": "image_url",
                        "image_url": ["url": "data:\(mediaType);base64,\(base64)"],
                    ])
                case .content(let block):
                    switch block {
                    // No portable way to replay reasoning state; dropping it is correct.
                    case .thinking, .redactedThinking, .openAIReasoning:
                        continue
                    case .text(let value):
                        text += value
                    case .toolUse(let id, let name, let inputJSON):
                        toolCalls.append([
                            "id": id,
                            "type": "function",
                            "function": ["name": name, "arguments": nonEmptyJSONObject(inputJSON)],
                        ])
                    case .toolResult(let toolUseID, let output, let isError):
                        if !text.isEmpty || !parts.isEmpty || !toolCalls.isEmpty {
                            out.append(turn(role: role, text: text, parts: parts, toolCalls: toolCalls))
                            text = ""
                            parts = []
                            toolCalls = []
                        }
                        out.append([
                            "role": "tool",
                            "tool_call_id": toolUseID,
                            "content": toolResultText(output, isError: isError),
                        ])
                    }
                }
            }
            guard !text.isEmpty || !parts.isEmpty || !toolCalls.isEmpty else { continue }
            out.append(turn(role: role, text: text, parts: parts, toolCalls: toolCalls))
        }
        return out
    }

    private static func turn(
        role: String,
        text: String,
        parts: [[String: Any]],
        toolCalls: [[String: Any]]
    ) -> [String: Any] {
        var message: [String: Any] = ["role": role]
        if parts.isEmpty {
            message["content"] = text
        } else {
            var content = parts
            if !text.isEmpty { content.insert(["type": "text", "text": text], at: 0) }
            message["content"] = content
        }
        if !toolCalls.isEmpty { message["tool_calls"] = toolCalls }
        return message
    }

    /// Servers reject an empty arguments string, and the agent's own history uses "{}" for
    /// a no-argument call, so a parse failure degrades to an empty object rather than bad JSON.
    private static func nonEmptyJSONObject(_ json: String) -> String {
        let trimmed = json.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let data = trimmed.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              JSONSerialization.isValidJSONObject(object),
              let normalized = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
              let string = String(data: normalized, encoding: .utf8)
        else { return "{}" }
        return string
    }

    private static func toolResultText(_ output: [ToolResult.Block], isError: Bool) -> String {
        let text = output.compactMap { block -> String? in
            if case .text(let value) = block { return value }
            return nil
        }.joined(separator: "\n")
        if isError { return text.isEmpty ? "Tool error" : "Tool error: \(text)" }
        return text
    }
}

enum OpenAIChatCompletionsSSE {
    static func parse(
        bytes: URLSession.AsyncBytes,
        continuation: AsyncThrowingStream<AgentStreamEvent, Error>.Continuation
    ) async throws {
        var parser = OpenAIChatCompletionsStreamParser()
        for try await line in bytes.lines {
            try Task.checkCancellation()
            for event in try parser.consume(line: line) {
                continuation.yield(event)
            }
        }
        try parser.finish()
    }
}

struct OpenAIChatCompletionsStreamParser {
    private struct PendingToolCall {
        var id = ""
        var name = ""
        var arguments = ""
    }

    private var pending: [Int: PendingToolCall] = [:]
    private var didTerminate = false

    mutating func consume(line: String) throws -> [AgentStreamEvent] {
        guard line.hasPrefix("data:") else { return [] }
        let payload = line.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
        guard payload != "[DONE]" else {
            return finishEvents()
        }
        guard let data = payload.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [] }
        if let error = object["error"] as? [String: Any] {
            throw AgentClientTransportError.streamError(
                provider: .custom,
                message: error["message"] as? String ?? "Unknown stream error."
            )
        }
        return try consume(chunk: object)
    }

    mutating func consume(chunk: [String: Any]) throws -> [AgentStreamEvent] {
        guard let choices = chunk["choices"] as? [[String: Any]],
              let choice = choices.first
        else { return [] }

        var events: [AgentStreamEvent] = []
        if let delta = choice["delta"] as? [String: Any] {
            if let reasoning = delta["reasoning_content"] as? String, !reasoning.isEmpty {
                events.append(.reasoningSummaryDelta(reasoning))
            }
            if let content = delta["content"] as? String, !content.isEmpty {
                events.append(.textDelta(content))
            }
            for call in delta["tool_calls"] as? [[String: Any]] ?? [] {
                accumulate(call)
            }
        }
        if let reason = choice["finish_reason"] as? String {
            events.append(contentsOf: finishEvents(finishReason: reason))
        }
        return events
    }

    private mutating func accumulate(_ call: [String: Any]) {
        // Servers key streamed calls by `index`; fall back to arrival order for those that don't.
        let slot = (call["index"] as? Int) ?? pending.count
        var entry = pending[slot] ?? PendingToolCall()
        if let id = call["id"] as? String, !id.isEmpty { entry.id = id }
        if let function = call["function"] as? [String: Any] {
            if let name = function["name"] as? String, !name.isEmpty { entry.name = name }
            if let arguments = function["arguments"] as? String { entry.arguments += arguments }
        }
        pending[slot] = entry
    }

    private mutating func finishEvents(finishReason: String? = nil) -> [AgentStreamEvent] {
        didTerminate = true
        var events: [AgentStreamEvent] = []
        for (_, call) in pending.sorted(by: { $0.key < $1.key }) where !call.name.isEmpty {
            events.append(.toolUseComplete(
                id: call.id.isEmpty ? "call_\(UUID().uuidString)" : call.id,
                name: call.name,
                inputJSON: call.arguments.isEmpty ? "{}" : call.arguments
            ))
        }
        pending.removeAll()
        events.append(.messageStop(stopReason: stopReason(for: finishReason)))
        return events
    }

    private func stopReason(for raw: String?) -> AgentStopReason {
        switch raw {
        case "tool_calls", "function_call": .toolUse
        case "length": .maxTokens
        case "content_filter": .refusal
        default: .endTurn
        }
    }

    func finish() throws {
        guard didTerminate else {
            throw AgentClientTransportError.streamError(
                provider: .custom,
                message: "The stream ended before a terminal event."
            )
        }
    }
}
