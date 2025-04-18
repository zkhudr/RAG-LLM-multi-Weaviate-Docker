// --- START static/ragapp.js ---

// Use alpine:init to ensure ragApp is defined before Alpine scans

let domContentLoadedFired = false; // <-- Add this flag outside listeners
document.addEventListener('alpine:init', () => {
    console.log('[Alpine Event] alpine:init fired. Defining ragApp component...');

    Alpine.data('ragApp', () => {
        console.log("[Alpine Data] Executing ragApp() definition..."); // Called once per component type definition

        return {
            // === STATE VARIABLES ===
            initCalled: false,
            statusMessage: 'Waiting for init...', // Initial state before explicit init
            userInput: '',
            chatHistory: [],
            filesToUpload: [],
            selectedFileNames: [],
            isStreaming: false,

            // --- Config State ---
            formConfig: { // Default structure, values loaded via API
                security: { SANITIZE_INPUT: true, RATE_LIMIT: 100, API_TIMEOUT: 30, CACHE_ENABLED: true },
                retrieval: { COLLECTION_NAME: '', K_VALUE: 5, SCORE_THRESHOLD: 0.5, LAMBDA_MULT: 0.5, SEARCH_TYPE: 'mmr', DOMAIN_SIMILARITY_THRESHOLD: 0.6, SPARSE_RELEVANCE_THRESHOLD: 0.1, FUSED_RELEVANCE_THRESHOLD: 0.4, SEMANTIC_WEIGHT: 0.6, SPARSE_WEIGHT: 0.4, PERFORM_DOMAIN_CHECK: true, WEAVIATE_HOST: '', WEAVIATE_HTTP_PORT: 0, WEAVIATE_GRPC_PORT: 0 },
                model: { LLM_TEMPERATURE: 0.7, MAX_TOKENS: 1024, OLLAMA_MODEL: '', EMBEDDING_MODEL: '', TOP_P: 1.0, FREQUENCY_PENALTY: 0.0, SYSTEM_MESSAGE: '', EXTERNAL_API_PROVIDER: 'none', EXTERNAL_API_MODEL_NAME: null },
                document: { CHUNK_SIZE: 1000, CHUNK_OVERLAP: 100, FILE_TYPES: [], PARSE_TABLES: true, GENERATE_SUMMARY: false },
                paths: { DOCUMENT_DIR: '', DOMAIN_CENTROID_PATH: '' },
                env: { DOMAIN_KEYWORDS: [], AUTO_DOMAIN_KEYWORDS: [], USER_ADDED_KEYWORDS: [] }
            },

            // --- Presets State ---
            presets: {}, // Loaded via API
            selectedPresetName: '',
            newPresetName: '',

            // --- Weaviate State ---
            weaviateInstances: [], // Loaded via API
            newInstanceName: '',

            // --- API Key Status ---
            apiKeyStatus: { deepseek: false, openai: false, anthropic: false, cohere: false }, // Loaded via API

            // --- Saved Chats State ---
            savedChats: [], // Loaded via API

            // --- UI State ---
            toast: { show: false, message: '', type: 'info', timeout: null },
            confirmationModal: { show: false, title: '', message: '', onConfirm: () => {}, onCancel: () => {}, confirmButtonClass: '' },

            // === INITIALIZATION METHOD (Called manually below) ===
            async init() {
                if (this.initCalled) { console.warn("[Init Method] init() called again, skipping."); return; }
                this.initCalled = true;
                console.log("[Init Method] init() explicitly called (First Run).");
                this.statusMessage = 'Loading UI data...';
                try {
                    console.log("[Init Method] Starting API calls in Promise.all...");
                    // Fetch initial data using Promise.all
                    await Promise.all([
                        (async () => { console.log("[Init API] Starting loadInitialConfig..."); try { await this.loadInitialConfig(); console.log("[Init API] Finished loadInitialConfig."); } catch(e){console.error("[Init API] loadInitialConfig FAILED:", e); throw e;} })(), // Re-throw critical errors
                        (async () => { console.log("[Init API] Starting checkApiKeys..."); try { await this.checkApiKeys(); console.log("[Init API] Finished checkApiKeys."); } catch(e){console.error("[Init API] checkApiKeys FAILED:", e); /* Non-critical? */ } })(),
                        (async () => { console.log("[Init API] Starting fetchPresets..."); try { await this.fetchPresets(); console.log("[Init API] Finished fetchPresets."); } catch(e){console.error("[Init API] fetchPresets FAILED:", e); /* Non-critical? */ } })(),
                        (async () => { console.log("[Init API] Starting fetchWeaviateInstances..."); try { await this.fetchWeaviateInstances(); console.log("[Init API] Finished fetchWeaviateInstances."); } catch(e){console.error("[Init API] fetchWeaviateInstances FAILED:", e); /* Non-critical? */ } })(),
                        (async () => { console.log("[Init API] Starting fetchSavedChats..."); try { await this.fetchSavedChats(); console.log("[Init API] Finished fetchSavedChats."); } catch(e){console.error("[Init API] fetchSavedChats FAILED:", e); /* Non-critical? */ } })()
                    ]);
                    console.log("[Init Method] Promise.all finished.");

                    // Add welcome message
                    if (this.chatHistory.length === 0) {
                         console.log("[Init Method] Adding welcome message.");
                        this.chatHistory.push({ role: 'assistant', text: 'Hello! Ask me anything.', timestamp: new Date().toISOString() });
                    } else {
                         console.log("[Init Method] Skipping welcome message (history already present).");
                    }

                    this.statusMessage = 'Idle';
                    console.log("[Init Method] init() finished successfully.");
                    this.scrollToBottom();
                    // Focus input after UI updates
                    this.$nextTick(() => {
                        console.log("[Init Method] Focusing input area.");
                        this.$refs.inputArea?.focus();
                    });

                } catch (error) {
                    // This catches errors specifically re-thrown from critical API calls (like loadInitialConfig)
                    console.error("[Init Method] CRITICAL Error during init:", error);
                    this.statusMessage = 'Initialization Error!';
                    this.showToast(`Initialization failed: ${error.message}. Check console & backend.`, 'error', 10000);
                }
            }, // End of init() method

            // === HELPER FUNCTIONS ===
            isLoading() {
                const msg = this.statusMessage.toLowerCase();
                return msg.includes('loading') || msg.includes('saving') || msg.includes('creating') ||
                       msg.includes('removing') || msg.includes('activating') || msg.includes('ingesting') ||
                       msg.includes('uploading') || msg.includes('sending') || msg.includes('replying') ||
                       msg.includes('applying'); // Add applying preset
            },
            isLoadingChat() { return this.isStreaming || this.statusMessage === 'Sending...' || this.statusMessage === 'Assistant is replying...'; },
            formatTimestamp(isoString) { if (!isoString) return ''; try { return new Date(isoString).toLocaleString(); } catch (e) { return isoString; } },
            scrollToBottom() { this.$nextTick(() => { const el = this.$refs.chatHistoryContainer; if (el) el.scrollTop = el.scrollHeight; }); },
            adjustTextareaHeight(el) { if (!el) el = this.$refs.inputArea; if (!el) return; const maxH = 150; el.style.height = 'auto'; el.style.height = `${Math.min(el.scrollHeight, maxH)}px`; },
            renderMarkdown(text) {
                if (!text) return '';
                // Basic: escape HTML, convert newlines to <br>, basic bold/italic
                let safeText = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                safeText = safeText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'); // Bold
                safeText = safeText.replace(/\*(.*?)\*/g, '<em>$1</em>');     // Italic
                safeText = safeText.replace(/`([^`]+)`/g, '<code>$1</code>'); // Inline code
                safeText = safeText.replace(/\n/g, '<br>');                     // Newlines
                // Add <pre><code> for blocks if needed, requires more complex regex or library
                return safeText;
            }, // Make sure comma is present if other methods follow

            showToast(message, type = 'info', duration = 3000) { console.log(`[Toast (${type})] ${message}`); this.toast.message = message; this.toast.type = type; this.toast.show = true; clearTimeout(this.toast.timeout); this.toast.timeout = setTimeout(() => { this.toast.show = false; }, duration); },
            requireConfirmation(options) {
                console.log(`[Confirm Req] Title: ${options.title || 'Confirmation'}, Message: ${options.message || 'Are you sure?'}`);
                // Ensure modal state exists (should be defined in initial state)
                if (!this.confirmationModal || typeof this.confirmationModal !== 'object') {
                    console.error("[requireConfirmation] ERROR: confirmationModal state object is missing!");
                    // Optionally show a generic toast if available
                    if (this.showToast) this.showToast("Cannot display confirmation dialog.", "error");
                    return; // Prevent further execution
                }
                // Assign properties
                this.confirmationModal.title = options.title || 'Confirmation';
                this.confirmationModal.message = options.message || 'Are you sure?';
                // Store the functions to be called later, ensuring they are functions
                this.confirmationModal.onConfirm = (typeof options.onConfirm === 'function') ? options.onConfirm : () => { console.warn("No valid onConfirm action provided for modal.") };
                this.confirmationModal.onCancel = (typeof options.onCancel === 'function') ? options.onCancel : () => { /* Default no-op is fine */ };
                this.confirmationModal.confirmButtonClass = options.confirmButtonClass || 'bg-blue-600 hover:bg-blue-700'; // Default style
                // Show the modal
                this.confirmationModal.show = true;
            },

            confirmAction() {
                console.log("[Modal Action] Confirm button clicked.");
                // Check if modal state and callback exist
                if (!this.confirmationModal || typeof this.confirmationModal.onConfirm !== 'function') {
                    console.error("[confirmAction] ERROR: Cannot execute confirmation, modal state or onConfirm callback is invalid.");
                    if (this.showToast) this.showToast("Confirmation action failed (internal error).", "error");
                    // Still hide the modal if possible
                    if (this.confirmationModal) this.confirmationModal.show = false;
                    return;
                }
                try {
                    // Execute the callback function stored when requireConfirmation was called
                    this.confirmationModal.onConfirm(); // Call the stored confirm logic
                } catch (error) {
                    console.error("Error executing confirmation action callback:", error);
                    // Use showToast if it exists
                    if (this.showToast) this.showToast(`Operation failed during confirmation: ${error.message}`, 'error');
                } finally {
                    // Always hide the modal after attempting the action
                    this.confirmationModal.show = false;
                    // Optional: Reset callbacks to prevent accidental reuse if needed
                    // this.confirmationModal.onConfirm = () => {};
                    // this.confirmationModal.onCancel = () => {};
                }
            },

            cancelAction() {
                console.log("[Modal Action] Cancel button clicked (or modal background clicked).");
                // Check if modal state and callback exist
                if (!this.confirmationModal) {
                    console.error("[cancelAction] ERROR: Cannot execute cancel, modal state is invalid.");
                    // Don't usually need a toast for cancel failure, just hide if possible
                    return;
                }
                try {
                    // Execute the callback function stored when requireConfirmation was called
                    if (typeof this.confirmationModal.onCancel === 'function') {
                        this.confirmationModal.onCancel(); // Call the stored cancel logic
                    }
                } catch (error) {
                    console.error("Error executing cancel action callback:", error);
                    // Rarely need user notification for cancel error
                } finally {
                    // Always hide the modal
                    this.confirmationModal.show = false;
                    // Optional: Reset callbacks
                    // this.confirmationModal.onConfirm = () => {};
                    // this.confirmationModal.onCancel = () => {};
                }
            },
            handleChatInputKeydown(event) { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); this.sendMessage(); } this.$nextTick(() => this.adjustTextareaHeight(event.target)); },
            handleFileSelect(event) { this.filesToUpload = Array.from(event.target.files || []); this.selectedFileNames = this.filesToUpload.map(f => f.name); },

            // === ASYNC ACTIONS (Backend Interaction - Implementations unchanged, ensure endpoints exist) ===
            // --- API Keys ---
            async checkApiKeys() {
                console.log("[API Call] Checking API keys via /api/key_status...");
                try {
                    const response = await fetch('/api/key_status'); // Ensure Flask has this GET endpoint
                    if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    const status = await response.json();
                    // Ensure structure matches default before assigning
                    this.apiKeyStatus.deepseek = status.deepseek || false;
                    this.apiKeyStatus.openai = status.openai || false;
                    this.apiKeyStatus.anthropic = status.anthropic || false;
                    this.apiKeyStatus.cohere = status.cohere || false;
                    console.log("[API Resp] API Key status loaded:", this.apiKeyStatus);
                } catch (error) {
                    console.error("[API Error] checkApiKeys FAILED:", error);
                    // Keep defaults (all false) on error
                    this.apiKeyStatus = { deepseek: false, openai: false, anthropic: false, cohere: false };
                    if (this.showToast) this.showToast('Could not check API key status.', 'error');
                }
            },

            // --- Weaviate Instance Management ---
            async fetchWeaviateInstances() {
                console.log("[API Call] Fetching Weaviate instances from /list_weaviate_instances...");
                this.statusMessage = 'Loading Weaviate instances...'; // Show loading status
                try {
                    const response = await fetch('/list_weaviate_instances'); // Ensure Flask has this GET endpoint
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    const data = await response.json();
                    console.log("[API Resp] Received Weaviate data:", data);
                    if (!Array.isArray(data)) throw new Error("Received invalid data format for instances.");

                    // Use the received list directly (active flag is set by backend)
                    this.weaviateInstances = data;

                    console.log("[API Logic] Processed Weaviate instances:", this.weaviateInstances);
                    // Only set status to Idle if this specific fetch succeeded
                    if (this.statusMessage === 'Loading Weaviate instances...') {
                        this.statusMessage = 'Idle';
                    }
                } catch (error) {
                    console.error("[API Error] fetchWeaviateInstances FAILED:", error);
                    if (this.showToast) this.showToast(`Error loading Weaviate instances: ${error.message}`, 'error');
                    this.weaviateInstances = []; // Clear list on error
                    this.statusMessage = 'Error loading instances'; // Keep error status
                }
            },

            async createWeaviateInstance() {
                // Added console log from previous step for debugging click
                console.log('[Button Click] Create button clicked. isLoading:', this.isLoading(), 'Instance Name:', this.newInstanceName);
                const instanceName = this.newInstanceName.trim();
                if (!instanceName) {
                    if (this.showToast) this.showToast("Please enter a name for the new instance.", "info");
                    return;
                }
                if (this.isLoading()) {
                    console.warn('[Button Click] Create ignored, already loading:', this.statusMessage);
                    return; // Prevent action if already busy
                }

                console.log(`[API Call] Creating Weaviate instance: ${instanceName}`);
                this.statusMessage = `Creating ${instanceName}...`;
                try {
                    const response = await fetch('/create_weaviate_instance', { // Ensure Flask POST endpoint exists
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ instance_name: instanceName })
                    });
                    const result = await response.json();
                    if (!response.ok) throw new Error(result.error || `HTTP error! Status: ${response.status}`);

                    if (this.showToast) this.showToast(result.message || `Instance '${instanceName}' creating...`, 'success');
                    this.newInstanceName = ''; // Clear input
                    await this.fetchWeaviateInstances(); // Refresh list after creation attempt

                } catch (error) {
                    console.error('[API Error] Create Instance FAILED:', error);
                    this.statusMessage = 'Create Error';
                    if (this.showToast) this.showToast(`Error creating instance: ${error.message}`, 'error');
                } finally {
                    // Reset status only if it wasn't changed by another operation in the meantime
                    if (this.statusMessage === `Creating ${instanceName}...`) {
                        setTimeout(() => { this.statusMessage = 'Idle'; }, 1000);
                    }
                }
            },

            // Uses confirmation modal
            removeWeaviateInstanceWithConfirmation(instanceName) {
                if (!instanceName || instanceName === "Default (from config)") {
                    if (this.showToast) this.showToast("Cannot remove the default configuration entry.", "info");
                    return;
                }
                if (this.isLoading()) {
                    console.warn('[Button Click] Remove ignored, already loading:', this.statusMessage);
                    return; // Prevent action if already busy
                }

                this.requireConfirmation({
                    title: 'Remove Weaviate Instance',
                    message: `Are you sure you want to remove instance "${instanceName}"? This will stop and delete its container and data volume.`,
                    confirmButtonClass: 'bg-red-600 hover:bg-red-700', // Red confirm button
                    onConfirm: async () => { // The actual async logic
                        console.log(`[Confirm Action] Removing Weaviate instance: ${instanceName}`);
                        this.statusMessage = `Removing ${instanceName}...`;
                        try {
                            const response = await fetch('/remove_weaviate_instance', { // Ensure Flask POST endpoint exists
                                method: 'POST', // Or DELETE if app.py uses DELETE
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ instance_name: instanceName })
                            });
                            const result = await response.json();
                            if (!response.ok) throw new Error(result.error || `HTTP error ${response.status}`);

                            if (this.showToast) this.showToast(result.message || `Instance '${instanceName}' removed.`, 'success');
                            // Refresh instances and potentially config after removal
                            await this.fetchWeaviateInstances();
                            await this.loadInitialConfig(); // Re-fetch config in case active instance changed

                        } catch (error) {
                            console.error('[API Error] Remove Instance FAILED:', error);
                            this.statusMessage = 'Remove Error';
                            if (this.showToast) this.showToast(`Error removing instance: ${error.message}`, 'error');
                        } finally {
                            if (this.statusMessage === `Removing ${instanceName}...`) {
                                setTimeout(() => { this.statusMessage = 'Idle'; }, 1000);
                            }
                        }
                    },
                    onCancel: () => { console.log(`[Confirm Action] Cancelled removal for: ${instanceName}`); }
                });
            },

            // Called by the 'Activate' button
            async activateRAG(instanceName) {
                if (!instanceName || this.isLoading()) {
                    if (!instanceName) console.warn('[Button Click] Activate ignored, no instance name.');
                    else console.warn('[Button Click] Activate ignored, already loading:', this.statusMessage);
                    return;
                }
                console.log(`[API Call] Activating instance: ${instanceName}`);
                this.statusMessage = `Activating ${instanceName}...`;
                try {
                    const response = await fetch('/select_weaviate_instance', { // Ensure Flask POST endpoint exists
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ instance_name: instanceName })
                    });
                    const result = await response.json();
                    if (!response.ok) throw new Error(result.error || `HTTP error! Status: ${response.status}`);

                    if (this.showToast) this.showToast(result.message || `Instance '${instanceName}' activated. Pipeline re-initializing...`, 'success');

                    // Refresh config (to get new host/port) and instance list (to show new active state)
                    await this.loadInitialConfig();
                    await this.fetchWeaviateInstances();

                    this.statusMessage = 'Idle'; // Set AFTER successful refresh

                } catch (error) {
                    console.error('[API Error] Activate Instance FAILED:', error);
                    this.statusMessage = 'Activation Error';
                    if (this.showToast) this.showToast(`Error activating instance: ${error.message}`, 'error');
                    setTimeout(() => { if (this.statusMessage === `Activating ${instanceName}...` || this.statusMessage === 'Activation Error') this.statusMessage = 'Idle'; }, 3000);
                }
            },

            // --- Chat Functionality ---
            async sendMessage() {
                const queryToSend = this.userInput.trim();
                if (!queryToSend) {
                    if (this.showToast) this.showToast("Please enter a message.", "info", 2000);
                    return;
                }
                if (this.isLoadingChat()) {
                    console.warn('[Chat Action] Send ignored, already processing:', this.statusMessage);
                    return;
                }

                // Add user message immediately
                this.chatHistory.push({ role: 'user', text: queryToSend, timestamp: new Date().toISOString() });
                this.userInput = ''; // Clear input
                this.adjustTextareaHeight(); // Adjust height after clearing
                this.scrollToBottom();

                this.isStreaming = true; // Indicate loading/streaming state
                this.statusMessage = 'Assistant is replying...';
                console.log(`[API Call] Sending query to /run_pipeline: ${queryToSend.substring(0, 50)}...`);

                try {
                    const response = await fetch('/run_pipeline', { // Ensure Flask POST endpoint accepts JSON
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: queryToSend })
                    });

                    if (!response.ok) {
                        let errorText = `Error: ${response.statusText} (Code: ${response.status})`;
                        try {
                            const errorJson = await response.json();
                            errorText = errorJson.error || errorJson.text || errorText; // Use error or text from backend response
                        } catch (e) { /* ignore if response isn't JSON */ }
                        throw new Error(errorText);
                    }

                    const result = await response.json(); // Expect { role: 'assistant', text: '...', sources: [...], error: false, timestamp: '...' }
                    console.log("[API Resp] Received from /run_pipeline:", result);

                    // Add assistant response (or error from backend)
                    this.chatHistory.push({
                        role: 'assistant',
                        text: result.text || result.response || "[No response content]",
                        sources: result.sources, // Attach sources if backend provides them
                        timestamp: result.timestamp || new Date().toISOString(), // Use backend timestamp if provided
                        error: result.error || false
                    });

                } catch (error) {
                    console.error('[API Error] Send Message FAILED:', error);
                    // Add error message to chat history
                    this.chatHistory.push({ role: 'assistant', text: `Sorry, an error occurred: ${error.message}`, timestamp: new Date().toISOString(), error: true });
                    if (this.showToast) this.showToast(`Error: ${error.message}`, 'error', 5000);
                } finally {
                    this.isStreaming = false; // End streaming state
                    this.statusMessage = 'Idle';
                    this.scrollToBottom();
                    this.$refs.inputArea?.focus(); // Re-focus input
                }
            },

            // Uses confirmation modal
            clearChatWithConfirmation() {
                if (this.isLoading() || !Array.isArray(this.chatHistory) || this.chatHistory.length === 0) return;
                this.requireConfirmation({
                    title: 'Clear Chat',
                    message: 'Are you sure you want to clear the current chat history?',
                    confirmButtonClass: 'bg-red-600 hover:bg-red-700',
                    onConfirm: async () => { // Keep async in case we add backend call later
                        console.log("[Confirm Action] Clearing chat history.");
                        this.statusMessage = 'Clearing chat...';
                        try {
                            // Optional: Backend call if session needs clearing server-side
                            // await fetch('/clear_chat_session', { method: 'POST' });
                            this.chatHistory = [{ role: 'assistant', text: 'Chat cleared. Ask me anything!', timestamp: new Date().toISOString() }]; // Reset with a message
                            if (this.showToast) this.showToast('Chat history cleared.', 'success');
                            this.scrollToBottom();
                        } catch (error) {
                            console.error("[Confirm Action Error] Clear Chat FAILED:", error);
                            if (this.showToast) this.showToast(`Error clearing chat: ${error.message}`, 'error');
                        } finally {
                            this.statusMessage = 'Idle';
                        }
                    },
                    onCancel: () => { console.log(`[Confirm Action] Cancelled chat clear.`); }
                });
            },

            // --- Saved Chats ---
            async fetchSavedChats() {
                console.log("[API Call] Fetching saved chats from /list_chats...");
                try {
                    const response = await fetch('/list_chats'); // Ensure Flask GET endpoint exists
                    if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    const loadedChats = await response.json();
                    this.savedChats = Array.isArray(loadedChats) ? loadedChats : []; // Ensure it's an array
                    console.log("[API Resp] Saved chats loaded:", this.savedChats.length);
                } catch (error) {
                    console.error("[API Error] fetchSavedChats FAILED:", error);
                    this.savedChats = []; // Ensure array on error
                    if (this.showToast) this.showToast(`Error loading saved chat list: ${error.message}`, 'error');
                }
            },

            async saveChat() {
                if (this.isLoading() || !Array.isArray(this.chatHistory) || this.chatHistory.length === 0) {
                    if (this.showToast) this.showToast("Nothing to save in current chat.", "info");
                    return;
                }
                // Use prompt for simplicity, could be replaced with a modal input
                const chatName = prompt("Enter a name for this chat:", `Chat ${new Date().toLocaleDateString()}`);
                if (!chatName) return; // User cancelled

                console.log(`[API Call] Saving current chat as: ${chatName}`);
                this.statusMessage = 'Saving chat...';
                try {
                    const response = await fetch('/save_chat', { // Ensure Flask POST endpoint accepts JSON
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        // Send name and the current history state
                        body: JSON.stringify({ name: chatName, history: this.chatHistory })
                    });
                    const result = await response.json();
                    if (!response.ok || !result.success) throw new Error(result.error || `HTTP error! Status: ${response.status}`);

                    if (this.showToast) this.showToast(result.message || `Chat '${chatName}' saved.`, 'success');
                    await this.fetchSavedChats(); // Refresh list

                } catch (error) {
                    console.error('[API Error] Save Chat FAILED:', error);
                    this.statusMessage = 'Save Error';
                    if (this.showToast) this.showToast(`Error saving chat: ${error.message}`, 'error');
                } finally {
                    if (this.statusMessage === 'Saving chat...') {
                        setTimeout(() => { this.statusMessage = 'Idle'; }, 1000);
                    }
                }
            },

            // Uses confirmation modal
            loadChatWithConfirmation(chatId) {
                if (!chatId || this.isLoading()) return;
                const chatToLoad = this.savedChats.find(c => c.id === chatId);
                if (!chatToLoad) { console.error("Chat ID not found in loaded list:", chatId); return; }

                this.requireConfirmation({
                    title: 'Load Chat',
                    message: `Load chat "${chatToLoad.name || chatId}"? This will replace the current chat history.`,
                    onConfirm: async () => {
                        console.log(`[Confirm Action] Loading chat: ${chatId}`);
                        this.statusMessage = 'Loading chat...';
                        try {
                            const response = await fetch(`/load_chat/${chatId}`); // Ensure Flask GET endpoint exists
                            if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                            const loadedChat = await response.json(); // Expect { id: ..., name: ..., history: [...] }

                            if (!Array.isArray(loadedChat.history)) throw new Error("Loaded chat history is not valid.");

                            this.chatHistory = loadedChat.history; // Replace current history
                            if (this.showToast) this.showToast(`Chat "${loadedChat.name || chatId}" loaded.`, 'success');
                            this.scrollToBottom();

                        } catch (error) {
                            console.error('[API Error] Load Chat FAILED:', error);
                            this.statusMessage = 'Load Error';
                            if (this.showToast) this.showToast(`Error loading chat: ${error.message}`, 'error');
                        } finally {
                            this.statusMessage = 'Idle';
                        }
                    },
                    onCancel: () => { console.log(`[Confirm Action] Cancelled load chat: ${chatId}`); }
                });
            },

            // Uses confirmation modal
            deleteChatWithConfirmation(chatId) {
                if (!chatId || this.isLoading()) return;
                const chatToDelete = this.savedChats.find(c => c.id === chatId);
                if (!chatToDelete) { console.error("Chat ID not found in loaded list:", chatId); return; }

                this.requireConfirmation({
                    title: 'Delete Saved Chat',
                    message: `Are you sure you want to permanently delete the saved chat "${chatToDelete.name || chatId}"?`,
                    confirmButtonClass: 'bg-red-600 hover:bg-red-700',
                    onConfirm: async () => {
                        console.log(`[Confirm Action] Deleting chat: ${chatId}`);
                        this.statusMessage = 'Deleting chat...';
                        try {
                            const response = await fetch(`/delete_chat/${chatId}`, { method: 'DELETE' }); // Ensure Flask DELETE endpoint exists
                            const result = await response.json();
                            if (!response.ok || !result.success) throw new Error(result.error || `HTTP error! Status: ${response.status}`);

                            if (this.showToast) this.showToast(result.message || 'Chat deleted.', 'success');
                            await this.fetchSavedChats(); // Refresh list

                        } catch (error) {
                            console.error('[API Error] Delete Chat FAILED:', error);
                            this.statusMessage = 'Delete Error';
                            if (this.showToast) this.showToast(`Error deleting chat: ${error.message}`, 'error');
                        } finally {
                            if (this.statusMessage === 'Deleting chat...') {
                                setTimeout(() => { this.statusMessage = 'Idle'; }, 1000);
                            }
                        }
                    },
                    onCancel: () => { console.log(`[Confirm Action] Cancelled delete chat: ${chatId}`); }
                });
            },

            // --- File Handling & Ingestion ---
            async uploadFiles() {
                // Added console log from previous step for debugging click
                console.log('[Button Click] Upload files clicked. isLoading:', this.isLoading(), 'File Count:', this.filesToUpload.length);
                if (!this.filesToUpload.length) {
                    if (this.showToast) this.showToast("No files selected to upload.", "info");
                    return;
                }
                if (this.isLoading()) {
                    console.warn('[Button Click] Upload ignored, already loading:', this.statusMessage);
                    return;
                }

                console.log(`[API Call] Uploading ${this.filesToUpload.length} files...`);
                this.statusMessage = 'Uploading files...';
                const formData = new FormData();
                this.filesToUpload.forEach(file => formData.append('files', file)); // 'files' must match Flask request.files.getlist key

                try {
                    const response = await fetch('/upload_files', { method: 'POST', body: formData }); // Ensure Flask POST endpoint exists
                    const result = await response.json();
                    if (!response.ok || !result.success) throw new Error(result.error || `HTTP error! Status: ${response.status}`);

                    if (this.showToast) this.showToast(`Files uploaded: ${result.files?.join(', ') || 'OK'}`, 'success');
                    this.filesToUpload = []; // Clear selection state
                    this.selectedFileNames = [];
                    // Clear the actual file input element visually
                    const fileInput = document.getElementById('fileUpload'); // Use the correct ID from index.html
                    if (fileInput) fileInput.value = '';

                } catch (error) {
                    console.error('[API Error] Upload Files FAILED:', error);
                    this.statusMessage = 'Upload Error';
                    if (this.showToast) this.showToast(`Upload failed: ${error.message}`, 'error');
                } finally {
                    if (this.statusMessage === 'Uploading files...') {
                        setTimeout(() => { this.statusMessage = 'Idle'; }, 1500);
                    }
                }
            },

            async startIngestion() {
                // Added console log from previous step for debugging click
                console.log('[Button Click] Start Full Ingestion clicked. isLoading:', this.isLoading());
                if (this.isLoading()) {
                    console.warn('[Button Click] Ingestion ignored, already loading:', this.statusMessage);
                    return;
                }
                console.log("[API Call] Starting full ingestion via /start_ingestion...");
                this.statusMessage = 'Starting Full Ingestion...'; // User feedback
                try {
                    const response = await fetch('/start_ingestion', { method: 'POST' }); // Ensure Flask POST endpoint exists
                    const result = await response.json();
                    if (!response.ok) { // Handle 503 or other errors
                        let errorDetail = result.error || `HTTP error! Status: ${response.status}`;
                        if (result.traceback) console.error("Ingestion Traceback:\n", result.traceback);
                        throw new Error(errorDetail);
                    }
                    // Process success response
                    let successMsg = result.message || 'Full ingestion finished.';
                    if (result.stats) successMsg += ` (Processed: ${result.stats.processed_chunks || 'N/A'}, Inserted: ${result.stats.inserted || 'N/A'}, Errors: ${result.stats.errors || 0})`;
                    if (this.showToast) this.showToast(successMsg, 'success', 8000); // Longer toast
                    this.statusMessage = 'Ingestion Complete';

                } catch (error) {
                    console.error('[API Error] Start Full Ingestion FAILED:', error);
                    this.statusMessage = 'Ingestion Error';
                    if (this.showToast) this.showToast(`Ingestion failed: ${error.message}`, 'error', 10000);
                } finally {
                    // Reset status after a delay, unless another operation started
                    setTimeout(() => { if (this.statusMessage === 'Starting Full Ingestion...' || this.statusMessage === 'Ingestion Complete' || this.statusMessage === 'Ingestion Error') this.statusMessage = 'Idle'; }, 4000);
                }
            },

            async startIncrementalIngestion() {
                // Added console log from previous step for debugging click
                console.log('[Button Click] Start Incremental Ingestion clicked. isLoading:', this.isLoading());
                if (this.isLoading()) {
                    console.warn('[Button Click] Incremental Ingestion ignored, already loading:', this.statusMessage);
                    return;
                }
                console.log("[API Call] Starting incremental ingestion via /ingest_block...");
                this.statusMessage = 'Starting Incremental Ingestion...';
                try {
                    const response = await fetch('/ingest_block', { method: 'POST' }); // Ensure Flask POST endpoint exists
                    const result = await response.json();
                    if (!response.ok) throw new Error(result.error || `HTTP error! Status: ${response.status}`);

                    let successMsg = result.message || 'Incremental ingestion finished.';
                    if (result.stats) successMsg += ` (Processed: ${result.stats.processed_chunks || 'N/A'}, Inserted: ${result.stats.inserted || 'N/A'}, Errors: ${result.stats.errors || 0})`;
                    if (this.showToast) this.showToast(successMsg, 'success', 6000);
                    this.statusMessage = 'Ingestion Complete';

                } catch (error) {
                    console.error('[API Error] Start Incremental Ingestion FAILED:', error);
                    this.statusMessage = 'Ingestion Error';
                    if (this.showToast) this.showToast(`Incremental ingestion failed: ${error.message}`, 'error', 10000);
                } finally {
                    setTimeout(() => { if (this.statusMessage === 'Starting Incremental Ingestion...' || this.statusMessage === 'Ingestion Complete' || this.statusMessage === 'Ingestion Error') this.statusMessage = 'Idle'; }, 4000);
                }
            },
            async loadInitialConfig() {
                console.log("Fetching initial config from /get_config...");
                try {
                    const response = await fetch('/get_config'); // MUST EXIST IN FLASK
                    if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    const loadedConfig = await response.json();

                    // Update formConfig state carefully, ensuring structure matches
                    for (const section in loadedConfig) {
                        if (this.formConfig[section] && typeof this.formConfig[section] === 'object') {
                            Object.assign(this.formConfig[section], loadedConfig[section]);
                        } else {
                            console.warn(`Section ${section} mismatch or not object.`);
                            this.formConfig[section] = loadedConfig[section];
                        }
                    }
                    // Ensure arrays are arrays
                    this.formConfig.document.FILE_TYPES = Array.isArray(this.formConfig.document.FILE_TYPES) ? this.formConfig.document.FILE_TYPES : [];
                    this.formConfig.env.DOMAIN_KEYWORDS = Array.isArray(this.formConfig.env.DOMAIN_KEYWORDS) ? this.formConfig.env.DOMAIN_KEYWORDS : [];
                    this.formConfig.env.AUTO_DOMAIN_KEYWORDS = Array.isArray(this.formConfig.env.AUTO_DOMAIN_KEYWORDS) ? this.formConfig.env.AUTO_DOMAIN_KEYWORDS : [];
                    this.formConfig.env.USER_ADDED_KEYWORDS = Array.isArray(this.formConfig.env.USER_ADDED_KEYWORDS) ? this.formConfig.env.USER_ADDED_KEYWORDS : [];

                    console.log("Initial config loaded successfully via API.");
                } catch (error) {
                    console.error("Failed to load initial config via API:", error);
                    this.showToast(`Error loading config: ${error.message}`, 'error');
                    // Keep default values defined in state
                    throw error; // Critical for init
                }
            },

            async fetchPresets() {
                console.log("Fetching presets from /list_presets...");
                try {
                    const response = await fetch('/list_presets'); // MUST EXIST IN FLASK
                    if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    this.presets = await response.json();
                    console.log("Presets loaded via API:", Object.keys(this.presets).length);
                } catch (error) {
                    console.error("Failed to fetch presets via API:", error);
                    this.presets = {};
                    this.showToast(`Error loading presets: ${error.message}`, 'error');
                    // Non-critical, don't re-throw
                }
            },

            async applyPreset(presetName) {
                // presetName comes from $event.target.value
                if (!presetName || this.isLoading()) return;
                console.log(`Requesting backend to apply preset: ${presetName}`);
                this.statusMessage = 'Applying Preset...';
                try {
                    // Option: Backend applies preset and returns the *new full config*
                    const response = await fetch(`/apply_preset/${presetName}`, { method: 'POST' }); // MUST EXIST IN FLASK
                    if (!response.ok) {
                        let errorData = { error: `HTTP error ${response.status}` };
                        try { errorData = await response.json(); } catch (e) { }
                        throw new Error(errorData.error || errorData.message || `Failed to apply preset.`);
                    }
                    const result = await response.json(); // Expect { success: true, config: { ... } }

                    if (!result.success || !result.config) {
                        throw new Error(result.message || "Backend didn't return updated config.");
                    }

                    // Update local formConfig with the config returned by the backend
                    for (const section in result.config) {
                        if (this.formConfig[section]) { Object.assign(this.formConfig[section], result.config[section]); }
                        else { this.formConfig[section] = result.config[section]; }
                    }
                    // Ensure arrays are correct after update
                    this.formConfig.document.FILE_TYPES = Array.isArray(this.formConfig.document.FILE_TYPES) ? this.formConfig.document.FILE_TYPES : [];
                    // ... repeat for other arrays ...

                    this.showToast(`Preset '${presetName}' applied.`, 'success');
                    this.selectedPresetName = presetName; // Ensure dropdown reflects applied preset
                    this.statusMessage = 'Idle';

                } catch (error) {
                    console.error('Apply Preset Error:', error);
                    this.statusMessage = 'Preset Error';
                    this.showToast(`Error applying preset: ${error.message}`, 'error');
                    this.selectedPresetName = ''; // Clear dropdown selection on error
                    setTimeout(() => { if (!this.isLoading()) this.statusMessage = 'Idle'; }, 3000);
                }
            },

            async savePreset() {
                const presetName = this.newPresetName.trim();
                // Validation: Check if name is provided
                if (!presetName) {
                    if (this.showToast) this.showToast("Enter a name for the new preset.", "info");
                    return; // Stop if no name
                }
                // Validation: Check if already performing an action
                if (this.isLoading()) {
                    console.warn('[Button Click] Save Preset ignored, already loading:', this.statusMessage);
                    if (this.showToast) this.showToast("Please wait for the current operation to finish.", "info");
                    return; // Stop if already busy
                }

                console.log(`[API Call] Saving current config as preset: ${presetName}`);
                this.statusMessage = 'Saving Preset...'; // Set loading state

                // Create a deep copy of the current config state to send
                // Use try/catch for JSON operations just in case formConfig is weird
                let configToSave;
                try {
                    configToSave = JSON.parse(JSON.stringify(this.formConfig));
                } catch (jsonError) {
                    console.error("[savePreset] Error preparing config data:", jsonError);
                    if (this.showToast) this.showToast("Internal error preparing configuration data.", "error");
                    this.statusMessage = 'Preset Error'; // Set error state
                    return; // Stop if data cannot be prepared
                }

                try {
                    // Make the API call to the backend
                    const response = await fetch('/save_preset', { // Ensure Flask POST endpoint exists and accepts JSON
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        // Send preset name and the config object
                        body: JSON.stringify({ preset_name: presetName, config: configToSave })
                    });

                    const result = await response.json(); // Always try to parse JSON response

                    // Check if the request was successful
                    if (!response.ok || !result.success) {
                        // Throw an error to be caught by the catch block
                        throw new Error(result.error || result.message || `HTTP error! Status: ${response.status}`);
                    }

                    // --- Success Path ---
                    if (this.showToast) this.showToast(result.message || `Preset '${presetName}' saved.`, 'success');
                    this.newPresetName = ''; // Clear the input field
                    await this.fetchPresets(); // Refresh the preset list dropdown
                    this.selectedPresetName = presetName; // Select the newly saved preset in the dropdown

                    // statusMessage will be reset in the finally block on success

                } catch (error) {
                    // --- Error Path ---
                    console.error('[API Error] Save Preset FAILED:', error);
                    this.statusMessage = 'Preset Error'; // Set specific error status
                    if (this.showToast) this.showToast(`Error saving preset: ${error.message}`, 'error');
                    // Optional: Delay before resetting error status to Idle
                    setTimeout(() => { if (this.statusMessage === 'Preset Error') this.statusMessage = 'Idle'; }, 3000);

                } finally {
                    // --- CORRECTED finally block ---
                    // Reset status *immediately* ONLY IF it's still 'Saving Preset...'
                    // This handles the success case correctly and avoids interfering with the error case.
                    if (this.statusMessage === 'Saving Preset...') {
                        this.statusMessage = 'Idle';
                        console.log("[savePreset finally] Status reset to Idle after success.");
                    } else {
                        // Log if status was already changed (e.g., by the catch block)
                        console.log("[savePreset finally] Status not 'Saving Preset...', current:", this.statusMessage);
                    }
                                  }
            }, // End of async savePreset
            async deletePresetWithConfirmation(presetName) {
                // Validation: Check if a preset name was provided (should come from selectedPresetName)
                if (!presetName) {
                    if (this.showToast) this.showToast("No preset selected to delete.", "info");
                    return;
                }
                // Validation: Check if already performing an action
                if (this.isLoading()) {
                    console.warn('[Button Click] Delete Preset ignored, already loading:', this.statusMessage);
                    if (this.showToast) this.showToast("Please wait for the current operation to finish.", "info");
                    return;
                }

                // Use the confirmation modal
                this.requireConfirmation({
                    title: 'Delete Preset',
                    message: `Are you sure you want to permanently delete the preset "${presetName}"?`,
                    confirmButtonClass: 'bg-red-600 hover:bg-red-700', // Red confirm button
                    onConfirm: async () => { // The actual deletion logic
                        console.log(`[Confirm Action] Deleting preset: ${presetName}`);
                        this.statusMessage = 'Deleting Preset...';
                        try {
                            // Make the API call to the backend
                            // Use encodeURIComponent in case preset names have special characters
                            const response = await fetch(`/delete_preset/${encodeURIComponent(presetName)}`, {
                                method: 'DELETE' // Use DELETE HTTP method
                            });

                            const result = await response.json(); // Expect { success: true/false, message: '...' }

                            // Check if the request was successful
                            if (!response.ok || !result.success) {
                                throw new Error(result.error || result.message || `HTTP error! Status: ${response.status}`);
                            }

                            // --- Success Path ---
                            if (this.showToast) this.showToast(result.message || `Preset '${presetName}' deleted.`, 'success');
                            await this.fetchPresets(); // Refresh the preset list dropdown
                            this.selectedPresetName = ''; // Clear the selection as the preset is gone

                            // statusMessage will be reset in finally block

                        } catch (error) {
                            // --- Error Path ---
                            console.error('[API Error] Delete Preset FAILED:', error);
                            this.statusMessage = 'Delete Error';
                            if (this.showToast) this.showToast(`Error deleting preset: ${error.message}`, 'error');
                            // Optional: Delay before resetting error status
                            setTimeout(() => { if (this.statusMessage === 'Delete Error') this.statusMessage = 'Idle'; }, 3000);

                        } finally {
                            // --- finally block ---
                            // Reset status ONLY IF it's still 'Deleting Preset...'
                            if (this.statusMessage === 'Deleting Preset...') {
                                this.statusMessage = 'Idle';
                                console.log("[deletePreset finally] Status reset to Idle after attempt.");
                            } else {
                                console.log("[deletePreset finally] Status not 'Deleting Preset...', current:", this.statusMessage);
                            }
                            // --- END finally block ---
                        }
                    }, // End onConfirm
                    onCancel: () => {
                        console.log(`[Confirm Action] Cancelled deletion for preset: ${presetName}`);
                    }
                }); // End requireConfirmation
            }, // End of deletePresetWithConfirmation

            // UPDATED: Sends JSON, relies on config.py's update_and_save via Flask endpoint
            async saveConfig() {
                if (this.isLoading()) return;
                console.log("Saving configuration via /save_config...");
                this.statusMessage = 'Saving Config...';
                // Create a deep copy of the current config state to send
                const configDataToSend = JSON.parse(JSON.stringify(this.formConfig));
                try {
                    const response = await fetch('/save_config', { // MUST EXIST IN FLASK (accepts JSON)
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(configDataToSend) // Send the whole config object
                    });
                    const result = await response.json(); // Expect { success: true/false, message: '...' }
                    if (!response.ok || !result.success) throw new Error(result.message || `HTTP error ${response.status}`);

                    this.showToast(result.message || 'Configuration saved successfully!', 'success');
                    // Optional: Re-fetch config if backend might modify it during save
                    // await this.loadInitialConfig();
                } catch (error) {
                    console.error('Configuration Save Error:', error);
                    this.statusMessage = 'Save Error';
                    this.showToast(`Error saving config: ${error.message}`, 'error');
                } finally {
                    // Reset status *immediately* ONLY IF it's still 'Saving Config...'
                    if (this.statusMessage === 'Saving Config...') {
                        this.statusMessage = 'Idle';
                        console.log("[saveConfig finally] Status reset to Idle after success.");
                    } else {
                        // Log if status was already changed (e.g., by the catch block)
                        console.log("[saveConfig finally] Status not 'Saving Config...', current:", this.statusMessage);
                    }
                }
            },

        }; // End of returned object
    }); // End of Alpine.data definition
}); // End of alpine:init listener

// Use DOMContentLoaded to explicitly initialize the tree and call init AFTER Alpine is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('[DOM Event] DOMContentLoaded fired.');
    // --- ADD THIS GUARD ---
    if (domContentLoadedFired) {
        console.warn('[DOM Event] DOMContentLoaded listener fired AGAIN. Preventing re-execution.');
        return; // Don't run the rest if already fired
    }
    domContentLoadedFired = true; // Set flag on first execution
    // --- END GUARD ---
    const rootElement = document.getElementById('rootAppComponent');
    if (rootElement) {
        console.log('[DOM Event] Found root element #rootAppComponent.');
        // Initialize Alpine components within the root element FIRST
        console.log('[DOM Event] Calling Alpine.initTree()...');
        Alpine.initTree(rootElement);
        console.log('[DOM Event] Alpine.initTree() finished.');

         // Wait a tick for Alpine to fully initialize components, then call init()
         Alpine.nextTick(() => {
             console.log('[DOM Event] Alpine.nextTick() executing...');
             // Access the component instance via the DOM element's data stack
             // _x_dataStack is internal, but usually reliable for root elements
             if (rootElement._x_dataStack && rootElement._x_dataStack.length > 0) {
                 const rootComponent = rootElement._x_dataStack[0]; // Access Alpine's data context
                 if (rootComponent && typeof rootComponent.init === 'function') {
                      console.log('[DOM Event] Calling rootComponent.init() explicitly...');
                      rootComponent.init(); // Call the init method
                 } else {
                      console.error('[DOM Event] Could not find root component instance or its init() method on _x_dataStack[0]!');
                 }
             } else {
                 console.error('[DOM Event] Alpine data stack (_x_dataStack) not found on root element!');
             }
         });
    } else {
        console.error('[DOM Event] Could not find root element with id="rootAppComponent"!');
    }
});

// Final log to confirm script parsing
console.log('static/ragapp.js parsed successfully.');

// --- END static/ragapp.js ---
