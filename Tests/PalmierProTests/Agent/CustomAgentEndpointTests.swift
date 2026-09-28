import Foundation
import Testing
@testable import PalmierPro

@Suite("Custom agent endpoint")
struct CustomAgentEndpointTests {

    @Test func baseURLNormalizationAcceptsTheFormsPeoplePaste() {
        #expect(CustomAgentEndpointStore.normalizedBaseURL("http://localhost:11434")?
            .absoluteString == "http://localhost:11434/v1")
        #expect(CustomAgentEndpointStore.normalizedBaseURL("http://localhost:11434/")?
            .absoluteString == "http://localhost:11434/v1")
        #expect(CustomAgentEndpointStore.normalizedBaseURL("http://localhost:11434/v1")?
            .absoluteString == "http://localhost:11434/v1")
        #expect(CustomAgentEndpointStore.normalizedBaseURL("https://api.example.com/openai/v1")?
            .absoluteString == "https://api.example.com/openai/v1")
        #expect(CustomAgentEndpointStore.normalizedBaseURL("ftp://example.com") == nil)
        #expect(CustomAgentEndpointStore.normalizedBaseURL("not a url") == nil)
    }

    @Test func onlyLoopbackHostsMayRunWithoutACredential() throws {
        func endpoint(_ raw: String) throws -> CustomAgentEndpoint {
            CustomAgentEndpoint(
                baseURL: try #require(CustomAgentEndpointStore.normalizedBaseURL(raw)),
                modelID: "m")
        }
        #expect(try endpoint("http://localhost:11434/v1").allowsMissingAPIKey)
        #expect(try endpoint("http://127.0.0.1:1234/v1").allowsMissingAPIKey)
        #expect(try endpoint("http://[::1]:1234/v1").allowsMissingAPIKey)
        #expect(try endpoint("https://api.example.com/v1").allowsMissingAPIKey == false)
    }

    @Test func chatCompletionsURLAppendsToTheVersionedBase() throws {
        let base = try #require(CustomAgentEndpointStore.normalizedBaseURL("http://localhost:11434"))
        let endpoint = CustomAgentEndpoint(baseURL: base, modelID: "qwen3:8b")
        #expect(endpoint.chatCompletionsURL.absoluteString == "http://localhost:11434/v1/chat/completions")
    }

    @Test func configuredEndpointRoutesDirectAndMissingOneIsUnavailable() throws {
        let local = CustomAgentEndpoint(
            baseURL: try #require(CustomAgentEndpointStore.normalizedBaseURL("http://localhost:11434")),
            modelID: "qwen3:8b")
        let remote = CustomAgentEndpoint(
            baseURL: try #require(CustomAgentEndpointStore.normalizedBaseURL("https://api.example.com/v1")),
            modelID: "some-model")

        #expect(AgentRouting.route(model: .custom, credentials: AgentCredentialSnapshot(),
                                  hasHostedCredits: true, hasPaidPlan: true, custom: local) == .direct)
        #expect(AgentRouting.route(model: .custom, credentials: AgentCredentialSnapshot(),
                                  hasHostedCredits: true, hasPaidPlan: true, custom: remote) == .unavailable)
        #expect(AgentRouting.route(model: .custom,
                                  credentials: AgentCredentialSnapshot([.custom: "key"]),
                                  hasHostedCredits: false, hasPaidPlan: false, custom: remote) == .direct)
        #expect(AgentRouting.route(model: .custom, credentials: AgentCredentialSnapshot([.custom: "key"]),
                                  hasHostedCredits: false, hasPaidPlan: false, custom: nil) == .unavailable)
    }

    @Test func runSettingsCarryTheEndpointOnlyForTheCustomModel() throws {
        let endpoint = CustomAgentEndpoint(
            baseURL: try #require(CustomAgentEndpointStore.normalizedBaseURL("http://localhost:11434/v1")),
            modelID: "qwen3:8b")
        #expect(AgentRunSettings(model: .custom, reasoningEffort: .none, custom: endpoint).custom == endpoint)
        #expect(AgentRunSettings(model: .terra, reasoningEffort: .medium, custom: endpoint).custom == nil)
    }

    @Test func requestBodyUsesTheConfiguredModelIDAndOmitsUnsupportedFields() throws {
        let body = chatBody(modelID: "qwen3-coder:30b")
        #expect(body["model"] as? String == "qwen3-coder:30b")
        #expect(body["stream"] as? Bool == true)
        // A fixed 64k budget and reasoning knobs are rejected by small local servers.
        #expect(body["max_tokens"] == nil)
        #expect(body["max_completion_tokens"] == nil)
        #expect(body["reasoning_effort"] == nil)
        #expect(body["store"] == nil)
    }

    @Test func requestBodyMapsToolsImagesAndToolResults() throws {
        let body = OpenAIChatCompletionsRequestBody.build(
            modelID: "m",
            system: "Instructions",
            tools: [AgentToolSchema(name: "inspect_timeline", description: "Inspect",
                                    inputSchema: ["type": "object"])],
            messages: [
                AgentRequestMessage(role: .user, content: [
                    .content(.text("Look")),
                    .image(base64: "aGVsbG8=", mediaType: "image/png"),
                ]),
                AgentRequestMessage(role: .assistant, content: [
                    .content(.thinking(text: "hidden", signature: "sig")),
                    .content(.text("Calling")),
                    .content(.toolUse(id: "call_1", name: "inspect_timeline", inputJSON: "{\"b\":2,\"a\":1}")),
                ]),
                AgentRequestMessage(role: .user, content: [
                    .content(.toolResult(toolUseId: "call_1", content: [.text("Done")], isError: false)),
                ]),
            ]
        )
        let messages = try #require(body["messages"] as? [[String: Any]])
        let tools = try #require(body["tools"] as? [[String: Any]])
        #expect(messages.first?["role"] as? String == "system")
        #expect(try #require(messages.first?["content"] as? String) == "Instructions")

        let user = try #require(messages.first { $0["role"] as? String == "user" })
        let parts = try #require(user["content"] as? [[String: Any]])
        #expect(parts.contains { $0["type"] as? String == "text" })
        #expect(parts.contains { $0["type"] as? String == "image_url" })

        let assistant = try #require(messages.first { $0["role"] as? String == "assistant" })
        #expect(assistant["content"] as? String == "Calling")
        let calls = try #require(assistant["tool_calls"] as? [[String: Any]])
        let function = try #require(calls.first?["function"] as? [String: Any])
        #expect(function["name"] as? String == "inspect_timeline")
        #expect(function["arguments"] as? String == "{\"a\":1,\"b\":2}")

        let tool = try #require(messages.first { $0["role"] as? String == "tool" })
        #expect(tool["tool_call_id"] as? String == "call_1")
        #expect(tool["content"] as? String == "Done")

        #expect(try #require((tools.first?["function"] as? [String: Any])?["name"] as? String)
            == "inspect_timeline")
        let data = try JSONSerialization.data(withJSONObject: body)
        let json = try #require(String(data: data, encoding: .utf8))
            .replacingOccurrences(of: "\\/", with: "/")
        #expect(json.contains("data:image/png;base64,aGVsbG8="))
        #expect(!json.contains("hidden"))
    }

    @Test func toolResultErrorsAreLabelledForTheModel() throws {
        let body = OpenAIChatCompletionsRequestBody.build(
            modelID: "m",
            system: "",
            tools: [],
            messages: [AgentRequestMessage(role: .user, content: [
                .content(.toolResult(toolUseId: "c1", content: [.text("bad path")], isError: true)),
            ])]
        )
        let messages = try #require(body["messages"] as? [[String: Any]])
        #expect(try #require(messages.first?["content"] as? String) == "Tool error: bad path")
    }

    @Test func parserReassemblesFragmentedToolCallsAndStopReason() throws {
        var parser = OpenAIChatCompletionsStreamParser()
        #expect(try parser.consume(line: #"data: {"choices":[{"delta":{"content":"Hi"}}]}"#)
            == [.textDelta("Hi")])
        #expect(try parser.consume(line: #"data: {"choices":[{"delta":{"reasoning_content":"think"}}]}"#)
            == [.reasoningSummaryDelta("think")])
        #expect(try parser.consume(line: #"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"inspect_timeline","arguments":"{\"a\":"}}]}}]}"#) == [])
        #expect(try parser.consume(line: #"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":"tool_calls"}]}"#)
            == [.toolUseComplete(id: "call_1", name: "inspect_timeline", inputJSON: "{\"a\":1}"),
                .messageStop(stopReason: .toolUse)])
        try parser.finish()
    }

    @Test func parserKeepsParallelToolCallsApart() throws {
        var parser = OpenAIChatCompletionsStreamParser()
        _ = try parser.consume(line: #"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"first","arguments":"{}"}}]}}]}"#)
        _ = try parser.consume(line: #"data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"b","function":{"name":"second","arguments":"{}"}}]}}]}"#)
        let events = try parser.consume(line: #"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#)
        #expect(events == [
            .toolUseComplete(id: "a", name: "first", inputJSON: "{}"),
            .toolUseComplete(id: "b", name: "second", inputJSON: "{}"),
            .messageStop(stopReason: .toolUse),
        ])
    }

    @Test func parserTerminatesOnDoneAndClassifiesFinishReasons() throws {
        var parser = OpenAIChatCompletionsStreamParser()
        #expect(try parser.consume(line: #"data: {"choices":[{"delta":{"content":"x"},"finish_reason":"length"}]}"#)
            == [.textDelta("x"), .messageStop(stopReason: .maxTokens)])
        try parser.finish()

        var done = OpenAIChatCompletionsStreamParser()
        #expect(try done.consume(line: "data: [DONE]") == [.messageStop(stopReason: .endTurn)])
        try done.finish()

        var filtered = OpenAIChatCompletionsStreamParser()
        #expect(try filtered.consume(line: #"data: {"choices":[{"finish_reason":"content_filter"}]}"#)
            == [.messageStop(stopReason: .refusal)])
    }

    @Test func parserSurfacesErrorsAndMissingTerminalEvent() {
        #expect(throws: AgentClientTransportError.self) {
            var parser = OpenAIChatCompletionsStreamParser()
            _ = try parser.consume(line: #"data: {"error":{"message":"model not found"}}"#)
        }
        #expect(throws: AgentClientTransportError.self) {
            try OpenAIChatCompletionsStreamParser().finish()
        }
    }

    private func chatBody(modelID: String) -> [String: Any] {
        AgentRunSettings(
            model: .custom,
            reasoningEffort: .none,
            custom: CustomAgentEndpoint(
                baseURL: URL(string: "http://localhost:11434/v1")!, modelID: modelID)
        ).requestBody(system: "Instructions", tools: [], messages: [])
    }
}

@Suite("Custom agent endpoint persistence")
@MainActor
struct CustomAgentEndpointPersistenceTests {

    @Test func endpointRoundTripsThroughDefaultsAndValidates() throws {
        try withDefaults { defaults in
            #expect(CustomAgentEndpointStore.save(baseURL: "", modelID: "m", defaults: defaults) != nil)
            #expect(CustomAgentEndpointStore.save(baseURL: "http://localhost:11434", modelID: "  ",
                                                  defaults: defaults) != nil)
            #expect(CustomAgentEndpointStore.save(baseURL: "localhost:11434", modelID: "m",
                                                  defaults: defaults) != nil)

            #expect(CustomAgentEndpointStore.save(baseURL: "http://localhost:11434/", modelID: "qwen3:8b",
                                                  defaults: defaults) == nil)
            let loaded = try #require(CustomAgentEndpointStore.load(defaults: defaults))
            #expect(loaded.baseURL.absoluteString == "http://localhost:11434/v1")
            #expect(loaded.modelID == "qwen3:8b")

            CustomAgentEndpointStore.clear(defaults: defaults)
            #expect(CustomAgentEndpointStore.load(defaults: defaults) == nil)
        }
    }

    @Test func customModelIsOfferedOnlyWhileAnEndpointExists() throws {
        try withDefaults { defaults in
            let service = AgentService(userDefaults: defaults)
            #expect(service.availableModels.contains(.custom) == false)
            #expect(service.canSelectModel(.custom) == false)

            #expect(CustomAgentEndpointStore.save(baseURL: "http://localhost:11434", modelID: "qwen3:8b",
                                                  defaults: defaults) == nil)
            service.updateCustomEndpoint(CustomAgentEndpointStore.load(defaults: defaults))
            #expect(service.availableModels.contains(.custom))
            #expect(service.canSelectModel(.custom))
            #expect(service.modelTitle(.custom) == "qwen3:8b")
            #expect(service.modelTitle(.terra) == "GPT-5.6 Terra")

            service.model = .custom
            #expect(service.snapshotRunSettings().custom?.modelID == "qwen3:8b")

            service.updateCustomEndpoint(nil)
            #expect(service.model == .defaultModel)
            #expect(service.availableModels.contains(.custom) == false)
        }
    }

    @Test func aSavedCustomModelFallsBackWhenNoEndpointRemains() throws {
        try withDefaults { defaults in
            #expect(CustomAgentEndpointStore.save(baseURL: "http://localhost:11434", modelID: "qwen3:8b",
                                                  defaults: defaults) == nil)
            _ = AgentService(userDefaults: defaults).model
            defaults.set("custom", forKey: "agentModel")
            CustomAgentEndpointStore.clear(defaults: defaults)

            let reopened = AgentService(userDefaults: defaults)
            #expect(reopened.model == .defaultModel)
        }
    }

    private func withDefaults(_ body: (UserDefaults) throws -> Void) throws {
        let suiteName = "CustomAgentEndpointTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }
        try body(defaults)
    }
}
