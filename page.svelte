<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { page } from "$app/stores";
    import { goto } from "$app/navigation";
    import { marked } from "marked";
    import {
        Send,
        Search,
        Users,
        Newspaper,
        FileText,
        Linkedin,
        Twitter,
        Github,
        Instagram,
        Loader2,
        ArrowUp,
        ArrowLeft,
    } from "lucide-svelte";

    marked.setOptions({
        breaks: true,
        gfm: true,
    });

    interface Message {
        role: "user" | "assistant" | "system";
        content: string;
        timestamp: number;
    }

    interface Step {
        step_number: number;
        status: "started" | "finished";
        tool_calls: { name: string; arguments: Record<string, any> }[];
        text: string;
        timestamp: number;
    }

    const API_BASE = "http://localhost:8081";

    let messages: Message[] = [];
    let steps: Step[] = [];
    let inputValue = "";
    let eventSource: EventSource | null = null;
    let isConnected = false;
    let isGenerating = false;
    let chatContainer: HTMLDivElement;
    let sessionId = "";
    let lastEventId = "0";

    function parseMarkdown(content: string): string {
        try {
            return marked(content) as string;
        } catch {
            return content;
        }
    }

    onMount(async () => {
        sessionId = $page.params.session_id;

        // Restore last event ID for resumption
        lastEventId = localStorage.getItem(`last_event_${sessionId}`) || "0";

        // Check session status before connecting
        await checkSessionStatus();
        connect();

        // Check for initial query from landing page
        const initialQuery = sessionStorage.getItem(
            `initial_query_${sessionId}`,
        );
        if (initialQuery) {
            sessionStorage.removeItem(`initial_query_${sessionId}`);
            // Wait for connection then send
            const waitForConnection = setInterval(() => {
                if (isConnected) {
                    clearInterval(waitForConnection);
                    sendInitialMessage(initialQuery);
                }
            }, 100);
        }
    });

    onDestroy(() => {
        if (eventSource) {
            eventSource.close();
        }
    });

    async function checkSessionStatus() {
        try {
            const response = await fetch(
                `${API_BASE}/agent/${sessionId}/state`,
            );
            // If we get state, session is active (or can be activated)
            if (response.ok) {
                // We can't determine "isGenerating" easily from just state existence
                // but we assume it's ready.
                // To check if it's generating, we'd need to inspect the state content if exposed
                const data = await response.json();
                // Assuming data.status or similar exists, but let's just confirm connectivity
                console.log("Session active");
            }
        } catch (e) {
            console.error("Failed to check session status:", e);
        }
    }

    function connect() {
        // Standard AgentServer SSE endpoint
        const url = `${API_BASE}/agent/${sessionId}/events?last_event_id=${lastEventId}`;
        eventSource = new EventSource(url);

        eventSource.onopen = () => {
            isConnected = true;
        };

        eventSource.onerror = () => {
            isConnected = false;
            // Reconnect after a delay
            setTimeout(() => {
                if (eventSource) {
                    eventSource.close();
                }
                connect();
            }, 3000);
        };

        // Handle different event types
        eventSource.addEventListener("connected", (e) => {
            isConnected = true;
            console.log("SSE connected");
        });

        eventSource.addEventListener("ai_start", (e) => {
            handleEvent("ai_start", JSON.parse(e.data), e.lastEventId);
        });

        eventSource.addEventListener("ai_chunk", (e) => {
            handleEvent("ai_chunk", JSON.parse(e.data), e.lastEventId);
        });

        eventSource.addEventListener("ai_end", (e) => {
            handleEvent("ai_end", JSON.parse(e.data), e.lastEventId);
        });

        eventSource.addEventListener("step_start", (e) => {
            handleEvent("step_start", JSON.parse(e.data), e.lastEventId);
        });

        eventSource.addEventListener("step_finish", (e) => {
            handleEvent("step_finish", JSON.parse(e.data), e.lastEventId);
        });

        eventSource.addEventListener("error", (e) => {
            if (e instanceof MessageEvent) {
                const data = JSON.parse(e.data);
                console.error("Server error:", data.error);
            }
        });

        eventSource.addEventListener("status", (e) => {
            handleEvent("status", JSON.parse(e.data), e.lastEventId);
        });
    }

    function handleEvent(type: string, data: any, eventId: string) {
        const timestamp = Date.now();

        // Store last event ID for reconnection
        if (eventId) {
            lastEventId = eventId;
            localStorage.setItem(`last_event_${sessionId}`, eventId);
        }

        const eventType = type;

        if (eventType === "ai_start") {
            isGenerating = true;
            steps = [];
            messages = [
                ...messages,
                { role: "assistant", content: "", timestamp },
            ];
        } else if (eventType === "ai_chunk") {
            const lastMsg = messages[messages.length - 1];
            if (lastMsg && lastMsg.role === "assistant") {
                lastMsg.content += data.content;
                messages = [...messages];
            }
        } else if (eventType === "ai_end") {
            isGenerating = false;
        } else if (eventType === "step_start") {
            const stepNum = data.step_number ?? data.details?.step_number;
            if (stepNum !== undefined) {
                steps = [
                    ...steps,
                    {
                        step_number: stepNum,
                        status: "started",
                        tool_calls: [],
                        text: "",
                        timestamp,
                    },
                ];
            }
        } else if (eventType === "step_finish") {
            const stepNum = data.step_number ?? data.details?.step_number;
            const toolCalls = data.tool_calls ?? data.details?.tool_calls ?? [];
            const text = data.text ?? data.details?.text ?? "";

            if (stepNum !== undefined) {
                const stepIndex = steps.findIndex(
                    (s) => s.step_number === stepNum,
                );
                if (stepIndex >= 0) {
                    steps[stepIndex] = {
                        ...steps[stepIndex],
                        status: "finished",
                        tool_calls: toolCalls,
                        text: text,
                    };
                    steps = [...steps];
                } else {
                    steps = [
                        ...steps,
                        {
                            step_number: stepNum,
                            status: "finished",
                            tool_calls: toolCalls,
                            text: text,
                            timestamp,
                        },
                    ];
                }
            }
        } else if (eventType === "status") {
            // Handle AgentOutput format
            const details = data.details || {};
            if (data.status === "step_start") {
                steps = [
                    ...steps,
                    {
                        step_number: details.step_number,
                        status: "started",
                        tool_calls: [],
                        text: "",
                        timestamp,
                    },
                ];
            } else if (data.status === "step_finish") {
                const stepIndex = steps.findIndex(
                    (s) => s.step_number === details.step_number,
                );
                if (stepIndex >= 0) {
                    steps[stepIndex] = {
                        ...steps[stepIndex],
                        status: "finished",
                        tool_calls: details.tool_calls || [],
                        text: details.text || "",
                    };
                    steps = [...steps];
                } else {
                    steps = [
                        ...steps,
                        {
                            step_number: details.step_number,
                            status: "finished",
                            tool_calls: details.tool_calls || [],
                            text: details.text || "",
                            timestamp,
                        },
                    ];
                }
            }
        }

        scrollToBottom();
    }

    async function sendInitialMessage(message: string) {
        messages = [
            ...messages,
            { role: "user", content: message, timestamp: Date.now() },
        ];
        scrollToBottom();

        try {
            // Use standard AgentServer chat endpoint
            const response = await fetch(
                `${API_BASE}/agent/${sessionId}/chat`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ message }),
                },
            );

            if (!response.ok) {
                // If chat endpoint doesn't exist, try invoke
                if (response.status === 404) {
                    const invokeResponse = await fetch(
                        `${API_BASE}/agent/${sessionId}/invoke`,
                        {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ action: "chat", message }),
                        },
                    );
                    if (!invokeResponse.ok) {
                        console.error(
                            "Failed to send message:",
                            await invokeResponse.text(),
                        );
                    }
                } else {
                    const error = await response.json();
                    console.error("Failed to send message:", error);
                }
            }
        } catch (e) {
            console.error("Failed to send message:", e);
        }
    }

    async function sendMessage() {
        if (!inputValue.trim() || !isConnected) return;

        const message = inputValue.trim();
        messages = [{ role: "user", content: message, timestamp: Date.now() }];
        inputValue = "";
        scrollToBottom();

        try {
            // Use standard AgentServer chat endpoint
            const response = await fetch(
                `${API_BASE}/agent/${sessionId}/chat`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ message }),
                },
            );

            if (!response.ok) {
                // If chat endpoint doesn't exist, try invoke
                if (response.status === 404) {
                    const invokeResponse = await fetch(
                        `${API_BASE}/agent/${sessionId}/invoke`,
                        {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ action: "chat", message }),
                        },
                    );
                    if (!invokeResponse.ok) {
                        console.error(
                            "Failed to send message:",
                            await invokeResponse.text(),
                        );
                    }
                } else {
                    const error = await response.json();
                    console.error("Failed to send message:", error);
                }
            }
        } catch (e) {
            console.error("Failed to send message:", e);
        }
    }

    function handleKeydown(e: KeyboardEvent) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    }

    function scrollToBottom() {
        setTimeout(() => {
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }, 10);
    }

    function goHome() {
        goto("/");
    }

    // Check if currently generating dossier
    $: isGeneratingDossier =
        steps.some(
            (s) =>
                s.status === "started" &&
                s.tool_calls.some((tc) => tc.name === "generate_dossier"),
        ) ||
        (isGenerating &&
            steps.some((s) =>
                s.tool_calls.some((tc) => tc.name === "generate_dossier"),
            ));
</script>

<div class="app">
    <!-- Header -->
    <header class="header">
        <button class="back-btn" on:click={goHome}>
            <ArrowLeft size={18} />
            <span>New Search</span>
        </button>
        <div class="session-id">
            {#if !isConnected}
                <Loader2 size={12} class="spinning" />
            {:else}
                <span class="status-dot"></span>
            {/if}
            <span>{sessionId}</span>
        </div>
    </header>

    <div class="container" bind:this={chatContainer}>
        {#if messages.length === 0}
            <!-- Waiting State -->
            <div class="waiting-state">
                <Loader2 size={24} class="spinning" />
                <span>Connecting to session...</span>
            </div>
        {:else}
            <!-- Conversation -->
            <div class="conversation">
                {#each messages as message, i}
                    {#if message.role === "user"}
                        <div class="user-query">
                            <span>{message.content}</span>
                        </div>
                    {:else if message.role === "assistant"}
                        <!-- Step Chain -->
                        {#if i === messages.length - 1 && steps.length > 0}
                            <div class="step-chain">
                                {#each steps as step, stepIndex}
                                    <div
                                        class="step-item"
                                        class:active={step.status === "started"}
                                    >
                                        <div class="step-connector">
                                            <div
                                                class="step-dot"
                                                class:active={step.status ===
                                                    "started"}
                                                class:done={step.status ===
                                                    "finished"}
                                            ></div>
                                            {#if stepIndex < steps.length - 1}
                                                <div class="step-line"></div>
                                            {/if}
                                        </div>
                                        <div class="step-content">
                                            {#if step.status === "started"}
                                                <Loader2
                                                    size={12}
                                                    class="spinning"
                                                />
                                            {/if}
                                            <span class="step-label">
                                                {#if step.tool_calls.length > 0}
                                                    {step.tool_calls
                                                        .map((tc) =>
                                                            tc.name.replace(
                                                                /_/g,
                                                                " ",
                                                            ),
                                                        )
                                                        .join(", ")}
                                                {:else if step.status === "started"}
                                                    thinking...
                                                {:else}
                                                    step {step.step_number}
                                                {/if}
                                            </span>
                                            {#if step.tool_calls.length > 0 && step.tool_calls[0].arguments}
                                                <span class="step-args">
                                                    {#if step.tool_calls[0].arguments.query}
                                                        "{step.tool_calls[0]
                                                            .arguments.query}"
                                                    {:else if step.tool_calls[0].arguments.name}
                                                        "{step.tool_calls[0]
                                                            .arguments.name}"
                                                    {:else if step.tool_calls[0].arguments.platform}
                                                        {step.tool_calls[0]
                                                            .arguments.platform}
                                                    {:else if step.tool_calls[0].arguments.url}
                                                        {step.tool_calls[0].arguments.url.substring(
                                                            0,
                                                            40,
                                                        )}...
                                                    {/if}
                                                </span>
                                            {/if}
                                        </div>
                                    </div>
                                {/each}
                            </div>
                        {/if}

                        <!-- Generating Dossier Indicator -->
                        {#if isGeneratingDossier && i === messages.length - 1}
                            <div class="dossier-generating">
                                <div class="dossier-icon">
                                    <FileText size={20} />
                                </div>
                                <div class="dossier-info">
                                    <span class="dossier-title"
                                        >Generating Dossier</span
                                    >
                                    <span class="dossier-subtitle"
                                        >Compiling research into a comprehensive
                                        profile...</span
                                    >
                                </div>
                                <div class="dossier-loader">
                                    <Loader2 size={16} class="spinning" />
                                </div>
                            </div>
                        {/if}

                        <div class="result-content markdown-content">
                            {#if message.content}
                                {@html parseMarkdown(message.content)}
                            {:else if isGenerating}
                                <div class="generating">
                                    <div class="generating-dots">
                                        <span></span><span></span><span></span>
                                    </div>
                                </div>
                            {/if}
                        </div>
                    {:else if message.role === "system"}
                        <div class="system-message">
                            {@html parseMarkdown(message.content)}
                        </div>
                    {/if}
                {/each}
            </div>
        {/if}
    </div>

    <!-- Bottom Input -->
    <div class="bottom-input">
        <div class="input-box">
            <input
                type="text"
                bind:value={inputValue}
                on:keydown={handleKeydown}
                placeholder="Ask a follow-up question..."
                disabled={!isConnected || isGenerating}
            />
            <div class="input-actions">
                <div class="input-left">
                    <button class="chip active">
                        <Search size={14} />
                        <span>Deep Search</span>
                    </button>
                </div>
                <button
                    class="send-btn"
                    on:click={sendMessage}
                    disabled={!inputValue.trim() ||
                        !isConnected ||
                        isGenerating}
                >
                    <ArrowUp size={16} />
                </button>
            </div>
        </div>
    </div>
</div>

<style>
    :global(*) {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    :global(body) {
        font-family: "SF Mono", "Fira Code", "JetBrains Mono", "Consolas",
            monospace;
        background: #0a0a0a;
        color: #e8e6e3;
        -webkit-font-smoothing: antialiased;
    }

    .app {
        min-height: 100vh;
        display: flex;
        flex-direction: column;
    }

    .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 24px;
        border-bottom: 1px solid #1a1a1a;
        background: #0a0a0a;
        position: sticky;
        top: 0;
        z-index: 100;
    }

    .back-btn {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: transparent;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        color: #888;
        font-size: 13px;
        cursor: pointer;
        transition: all 0.15s;
        font-family: inherit;
    }

    .back-btn:hover {
        border-color: #444;
        color: #e8e6e3;
    }

    .session-id {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        color: #666;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background: #4ade80;
        border-radius: 50%;
    }

    .container {
        max-width: 900px;
        width: 100%;
        margin: 0 auto;
        padding: 0 24px;
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow-y: auto;
    }

    .waiting-state {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 16px;
        color: #666;
        font-size: 14px;
    }

    /* Conversation */
    .conversation {
        flex: 1;
        padding: 32px 0 180px;
    }

    .user-query {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 24px;
    }

    .user-query span {
        background: #1f1f1f;
        padding: 12px 20px;
        border-radius: 20px;
        font-size: 14px;
        max-width: 70%;
    }

    /* Step Chain */
    .step-chain {
        background: #111;
        border: 1px solid #222;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }

    .step-item {
        display: flex;
        gap: 12px;
    }

    .step-connector {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 12px;
    }

    .step-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #333;
        border: 2px solid #444;
        flex-shrink: 0;
    }

    .step-dot.active {
        background: #f5e6d3;
        border-color: #f5e6d3;
        box-shadow: 0 0 8px rgba(245, 230, 211, 0.4);
    }

    .step-dot.done {
        background: #4ade80;
        border-color: #4ade80;
    }

    .step-line {
        width: 2px;
        flex: 1;
        min-height: 20px;
        background: #333;
        margin: 4px 0;
    }

    .step-content {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        padding-bottom: 12px;
        flex: 1;
    }

    .step-item:last-child .step-content {
        padding-bottom: 0;
    }

    .step-label {
        font-size: 13px;
        color: #e8e6e3;
        text-transform: lowercase;
    }

    .step-item.active .step-label {
        color: #f5e6d3;
    }

    .step-args {
        font-size: 12px;
        color: #666;
        max-width: 300px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    /* Dossier Generating */
    .dossier-generating {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 20px 24px;
        background: linear-gradient(135deg, #1a1a1a 0%, #111 100%);
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        margin-bottom: 20px;
    }

    .dossier-icon {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, #f5e6d3 0%, #e8d5c4 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #0a0a0a;
    }

    .dossier-info {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .dossier-title {
        font-size: 14px;
        font-weight: 600;
        color: #f5e6d3;
    }

    .dossier-subtitle {
        font-size: 12px;
        color: #888;
    }

    .dossier-loader {
        color: #f5e6d3;
    }

    :global(.spinning) {
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }

    .result-content {
        margin-bottom: 24px;
    }

    .generating {
        padding: 20px 0;
    }

    .generating-dots {
        display: flex;
        gap: 6px;
    }

    .generating-dots span {
        width: 6px;
        height: 6px;
        background: #666;
        border-radius: 50%;
        animation: pulse 1.4s infinite ease-in-out both;
    }

    .generating-dots span:nth-child(1) {
        animation-delay: -0.32s;
    }
    .generating-dots span:nth-child(2) {
        animation-delay: -0.16s;
    }

    @keyframes pulse {
        0%,
        80%,
        100% {
            transform: scale(0.6);
            opacity: 0.4;
        }
        40% {
            transform: scale(1);
            opacity: 1;
        }
    }

    .system-message {
        padding: 16px 20px;
        background: #111;
        border-radius: 12px;
        margin-bottom: 24px;
        font-size: 13px;
        color: #888;
        line-height: 1.6;
    }

    /* Bottom Input */
    .bottom-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 24px;
        background: linear-gradient(transparent, #0a0a0a 30%);
    }

    .input-box {
        width: 100%;
        max-width: 680px;
        background: #141414;
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 16px 20px;
        margin: 0 auto;
    }

    .input-box input {
        width: 100%;
        background: transparent;
        border: none;
        outline: none;
        font-size: 15px;
        color: #e8e6e3;
        font-family: inherit;
        margin-bottom: 12px;
    }

    .input-box input::placeholder {
        color: #666;
    }

    .input-actions {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .input-left {
        display: flex;
        gap: 8px;
    }

    .chip {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        background: #1f1f1f;
        border: 1px solid #333;
        border-radius: 20px;
        color: #888;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.15s;
        font-family: inherit;
    }

    .chip:hover {
        background: #2a2a2a;
        color: #aaa;
    }

    .chip.active {
        background: #2a2a2a;
        border-color: #444;
        color: #e8e6e3;
    }

    .send-btn {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #f5e6d3;
        border: none;
        color: #0a0a0a;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.15s;
    }

    .send-btn:hover:not(:disabled) {
        background: #fff;
    }

    .send-btn:disabled {
        background: #333;
        color: #666;
        cursor: not-allowed;
    }

    /* Markdown Content */
    :global(.markdown-content) {
        font-size: 14px;
        line-height: 1.7;
        color: #d4d4d4;
    }

    :global(.markdown-content h1) {
        font-size: 20px;
        font-weight: 600;
        color: #f5f0e8;
        margin-top: 28px;
        margin-bottom: 16px;
    }

    :global(.markdown-content h1:first-child) {
        margin-top: 0;
    }

    :global(.markdown-content h2) {
        font-size: 16px;
        font-weight: 600;
        color: #e8e6e3;
        margin-top: 24px;
        margin-bottom: 12px;
    }

    :global(.markdown-content h3) {
        font-size: 14px;
        font-weight: 600;
        color: #e8e6e3;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    :global(.markdown-content p) {
        margin-bottom: 14px;
    }

    :global(.markdown-content ul),
    :global(.markdown-content ol) {
        margin-left: 20px;
        margin-bottom: 14px;
    }

    :global(.markdown-content li) {
        margin-bottom: 6px;
    }

    :global(.markdown-content strong) {
        font-weight: 600;
        color: #e8e6e3;
    }

    :global(.markdown-content a) {
        color: #f5e6d3;
        text-decoration: none;
    }

    :global(.markdown-content a:hover) {
        text-decoration: underline;
    }

    :global(.markdown-content code) {
        background: #1a1a1a;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 12px;
        color: #aaa;
    }

    :global(.markdown-content pre) {
        background: #111;
        border: 1px solid #222;
        padding: 16px;
        border-radius: 12px;
        overflow-x: auto;
        margin: 16px 0;
    }

    :global(.markdown-content pre code) {
        background: none;
        padding: 0;
    }

    :global(.markdown-content blockquote) {
        border-left: 2px solid #333;
        padding-left: 16px;
        margin: 14px 0;
        color: #888;
    }

    :global(.markdown-content table) {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        font-size: 13px;
    }

    :global(.markdown-content th),
    :global(.markdown-content td) {
        text-align: left;
        padding: 12px 16px;
        border-bottom: 1px solid #222;
    }

    :global(.markdown-content th) {
        font-weight: 600;
        color: #888;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    :global(.markdown-content tr:hover) {
        background: #111;
    }

    :global(.markdown-content hr) {
        border: none;
        border-top: 1px solid #222;
        margin: 24px 0;
    }

    /* Inline source badges */
    :global(.markdown-content a[href*="x.com"]),
    :global(.markdown-content a[href*="twitter.com"]),
    :global(.markdown-content a[href*="linkedin.com"]),
    :global(.markdown-content a[href*="github.com"]) {
        background: #1a1a1a;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        color: #888;
        text-decoration: none;
        margin-left: 4px;
    }

    :global(.markdown-content a[href*="x.com"]:hover),
    :global(.markdown-content a[href*="twitter.com"]:hover),
    :global(.markdown-content a[href*="linkedin.com"]:hover),
    :global(.markdown-content a[href*="github.com"]:hover) {
        background: #2a2a2a;
        color: #aaa;
        text-decoration: none;
    }
</style>
