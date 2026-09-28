import Foundation

extension Notification.Name {
    /// Posted by Settings after the custom endpoint is saved or cleared. The agent service is
    /// owned per editor, so it cannot be called directly from the settings pane.
    static let agentCustomEndpointChanged = Notification.Name("agentCustomEndpointChanged")
}

/// A user-supplied OpenAI-compatible chat endpoint. The agent speaks
/// `/chat/completions` to it, which is the dialect Ollama, LM Studio, vLLM, LiteLLM,
/// OpenRouter, Groq and most gateways implement.
struct CustomAgentEndpoint: Equatable, Sendable {
    /// Base URL including the version segment, e.g. `http://localhost:11434/v1`.
    let baseURL: URL
    /// Wire model identifier, e.g. `qwen3-coder:30b` or `anthropic/claude-sonnet-5`.
    let modelID: String

    var chatCompletionsURL: URL {
        baseURL.appending(path: "chat/completions")
    }

    /// Loopback servers (Ollama, LM Studio) need no credential; anything remote must present a key.
    var allowsMissingAPIKey: Bool {
        guard let host = baseURL.host?.lowercased() else { return false }
        return host == "localhost" || host == "127.0.0.1" || host == "::1" || host.hasSuffix(".localhost")
    }

    var displayTitle: String { modelID }
}

enum CustomAgentEndpointStore {
    private static let baseURLKey = "agentCustomEndpoint.baseURL"
    private static let modelIDKey = "agentCustomEndpoint.modelID"

    static func load(defaults: UserDefaults = .standard) -> CustomAgentEndpoint? {
        guard let raw = defaults.string(forKey: baseURLKey)?.trimmingCharacters(in: .whitespaces),
              !raw.isEmpty,
              let baseURL = normalizedBaseURL(raw),
              let modelID = defaults.string(forKey: modelIDKey)?.trimmingCharacters(in: .whitespaces),
              !modelID.isEmpty
        else { return nil }
        return CustomAgentEndpoint(baseURL: baseURL, modelID: modelID)
    }

    /// Returns the validation failure, or nil when the endpoint is usable.
    @MainActor
    @discardableResult
    static func save(
        baseURL raw: String,
        modelID rawModelID: String,
        defaults: UserDefaults = .standard
    ) -> String? {
        let base = raw.trimmingCharacters(in: .whitespaces)
        let modelID = rawModelID.trimmingCharacters(in: .whitespaces)
        guard !base.isEmpty, let url = normalizedBaseURL(base) else {
            return L10n.string("Enter a valid http:// or https:// base URL.")
        }
        guard !modelID.isEmpty else { return L10n.string("Enter a model identifier.") }
        defaults.set(url.absoluteString, forKey: baseURLKey)
        defaults.set(modelID, forKey: modelIDKey)
        return nil
    }

    static func clear(defaults: UserDefaults = .standard) {
        defaults.removeObject(forKey: baseURLKey)
        defaults.removeObject(forKey: modelIDKey)
    }

    /// Accepts a base with or without the version segment and strips a trailing slash, so
    /// `localhost:11434`, `localhost:11434/`, and `localhost:11434/v1` all mean the same thing.
    static func normalizedBaseURL(_ raw: String) -> URL? {
        let trimmed = raw.hasSuffix("/") ? String(raw.dropLast()) : raw
        guard let url = URL(string: trimmed), url.scheme == "http" || url.scheme == "https",
              url.host?.isEmpty == false
        else { return nil }
        guard url.path.isEmpty || url.path == "/" else { return url }
        return url.appending(path: "v1")
    }
}
