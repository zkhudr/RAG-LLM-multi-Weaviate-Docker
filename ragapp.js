// --- START static/ragapp.js ---
// Define Alpine component at the right time, no global ragApp, no DOMContentLoaded flag

// ragapp.js
document.addEventListener('alpine:init', () => {
    console.log('[Alpine Event] alpine:init fired. Defining ragApp component...');
    Alpine.data('ragApp', () => ({
        

        envUserKeywordsString: '',
        // ─── CORE STATE ───
        docFreqMode: 'absolute',   // which mode is active
        totalDocs: 0,              // fetched from server
        initCalled: false,
        statusMessage: 'Initializing...',
        userInput: '',
        isStreaming: false,
        newPresetName: '',
        newInstanceName: '',
        selectedPresetName: '',
        chatHistory: [],
        filesToUpload: [],
        selectedFileNames: [],
        weaviateInstances: [],
        savedChats: [],
        presets: {},
        apiKeyStatus: { deepseek: false, openai: false, anthropic: false, cohere: false },
        toast: { show: false, message: '', type: 'info', timeout: null },
        confirmationModal: { show: false, title: '', message: '', onConfirm: () => { }, onCancel: () => { }, confirmButtonClass: '' },
        autoDomainKeywords: '',
        lastAutoDomainKeywordsList: [],
        centroidStats: {},
        queryCentroidInsight: {},
        centroidUpdateMode: 'auto',
        centroidAutoThreshold: 5,

        // ─── CONFIGURATION ───
        formConfig: {

            pipeline: {},

            paths: {
                DOCUMENT_DIR: '',           // default or loaded from server
                DOMAIN_CENTROID_PATH: ''    // default or loaded from server
            },
        
            domain_keyword_extraction: {
                min_doc_freq_abs: null,
                min_doc_freq_frac: null,
            },
            env: {
                DOMAIN_KEYWORDS: [],
                AUTO_DOMAIN_KEYWORDS: [],
                USER_ADDED_KEYWORDS: [],
                // if you use it in the template, you can also default SELECTED_N_TOP here
                SELECTED_N_TOP: null
              },
            
            security: {
                SANITIZE_INPUT: true,
                RATE_LIMIT: 100,
                API_TIMEOUT: 30,
                CACHE_ENABLED: true,
            },
            retrieval: {
                COLLECTION_NAME: '',
                K_VALUE: 5,
                SCORE_THRESHOLD: 0.5,
                LAMBDA_MULT: 0.5,
                SEARCH_TYPE: 'mmr',
                DOMAIN_SIMILARITY_THRESHOLD: 0.6,
                SPARSE_RELEVANCE_THRESHOLD: 0.1,
                FUSED_RELEVANCE_THRESHOLD: 0.4,
                SEMANTIC_WEIGHT: 0.6,
                SPARSE_WEIGHT: 0.4,
                PERFORM_DOMAIN_CHECK: true,
                WEAVIATE_HOST: '',
                WEAVIATE_HTTP_PORT: 0,
                WEAVIATE_GRPC_PORT: 0,
            },
            model: {
                LLM_TEMPERATURE: 0.7,
                MAX_TOKENS: 1024,
                OLLAMA_MODEL: '',
                EMBEDDING_MODEL: '',
                TOP_P: 1.0,
                FREQUENCY_PENALTY: 0.0,
                SYSTEM_MESSAGE: '',
                EXTERNAL_API_PROVIDER: 'none',
                EXTERNAL_API_MODEL_NAME: null,
            },
            document: {
                CHUNK_SIZE: 1000,
                CHUNK_OVERLAP: 100,
                FILE_TYPES: [],
                PARSE_TABLES: true,
                GENERATE_SUMMARY: false,
            },
            // …any other config sections…
        },

        // ─── LIFECYCLE METHODS ───
        init() {
            if (this.initCalled) return;
            this.initCalled = true;

            // Watch abs → update frac if in absolute mode
            this.$watch(
                'formConfig.domain_keyword_extraction.min_doc_freq_abs',
                abs => {
                    console.log('ABS watcher', abs);
                    if (this.docFreqMode === 'absolute' && this.totalDocs) {
                        this.formConfig.domain_keyword_extraction.min_doc_freq_frac =
                            +(abs / this.totalDocs).toFixed(3);
                    }
                }
            );

            // Watch frac → update abs if in fraction mode
            this.$watch(
                'formConfig.domain_keyword_extraction.min_doc_freq_frac',
                frac => {
                    console.log('FRAC watcher', frac);
                    if (this.docFreqMode === 'fraction' && this.totalDocs) {
                        this.formConfig.domain_keyword_extraction.min_doc_freq_abs =
                            Math.ceil(frac * this.totalDocs);
                    }
                }
            );

            // Kick off data load
            this.loadInitialData();
        },

        async loadInitialData() {
            // Fetch totalDocs
            console.log("[Init API] Fetching document count...");
            try {
                const res = await fetch('/get-doc-count');
                const json = await res.json();
                this.totalDocs = json.total_docs;
                console.log("[Init API] Document count loaded:", this.totalDocs);
            } catch (e) {
                console.warn("[Init API] Could not fetch doc count", e);
                this.totalDocs = 0;
            }

            // Initialize both abs ↔ frac from whichever the server provided
            const dke = this.formConfig.domain_keyword_extraction;
            if (dke.min_doc_freq_abs != null) {
                dke.min_doc_freq_frac = this.totalDocs
                    ? +(dke.min_doc_freq_abs / this.totalDocs).toFixed(3)
                    : 0;
            } else if (dke.min_doc_freq_frac != null) {
                dke.min_doc_freq_abs = this.totalDocs
                    ? Math.ceil(dke.min_doc_freq_frac * this.totalDocs)
                    : 0;
            } else {
                dke.min_doc_freq_abs = 0;
                dke.min_doc_freq_frac = 0;
            }
        },

        // === Auto Domain Keywords Controls ===

            autoDomainKeywordsList: [],        // Currently active auto keywords
            allAutoDomainKeywordsList: [],     // Full fetched list
            lastAutoDomainKeywordsList: [],    // Backup for “restore”
            activeAutoDomainKeywordsSet: new Set(),
            autoKeywordsEnabled: true,
            topNOptions: [],                   // e.g. [10, 20, …]
            selectedTopN: 10000,

            // === Computed getters ===
            get topNOptionsHtml() {
                return this.topNOptions
                    .map(n => `<option value="${n}">${n}</option>`)
                    .join('');
            },

        get coloredAutoDomainKeywords() {
            // Show all candidates, blue if within Top N & enabled, red otherwise
            return (this.allAutoDomainKeywordsList || []).map((term, idx) => {
                const css = (!this.autoKeywordsEnabled)
                    ? 'text-gray-400'
                    : (idx < this.selectedTopN ? 'text-blue-600' : 'text-red-600');
                return `<span class="${css}">${term}</span>`;
            }).join(', ');
              },


        applyTopNKeywords() {
            console.log(`[TopN] Applying top ${this.selectedTopN} keywords`);
            // Save current for restore
            this.lastAutoDomainKeywordsList = [...this.autoDomainKeywordsList];
            // Truncate to Top N
            this.autoDomainKeywordsList = this.allAutoDomainKeywordsList.slice(0, this.selectedTopN);
            this._updateActiveSet();
            this.syncActiveKeywordsToBackend();
            // Persist selection
            this.saveTopNToConfig(this.selectedTopN);
              },

        async saveTopNToConfig(topN) {
            try {
                console.log(`[Config] Saving TopN=${topN} to config`);
                const response = await fetch('/update_topn_config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ topN })
                });
                const result = await response.json();
                if (!result.success) {
                    console.error("[Config] Failed to save TopN:", result.error);
                } else {
                    console.log("[Config] TopN saved successfully");
                }
            } catch (e) {
                console.error("[Config] Error saving TopN to config:", e);
            }
              },

            // === Toggle / Restore ===
        toggleAutoKeywords() {
            this.autoKeywordsEnabled = !this.autoKeywordsEnabled;
            console.log("[AutoKeywords] Enabled:", this.autoKeywordsEnabled);

            if (this.autoKeywordsEnabled) {
                // Re-enable: restore or refetch
                if (this.lastAutoDomainKeywordsList.length > 0) {
                    this.autoDomainKeywordsList = [...this.lastAutoDomainKeywordsList];
                    this._updateActiveSet();
                    this.syncActiveKeywordsToBackend();
                    this.showToast("Auto keywords enabled and restored.", "success");
                } else {
                    this.fetchAutoDomainKeywords().then(() => {
                        this.showToast("Auto keywords enabled.", "success");
                    });
                }
            } else {
                // Disable: stash current then clear
                this.lastAutoDomainKeywordsList = [...this.autoDomainKeywordsList];
                this.autoDomainKeywordsList = [];
                this._updateActiveSet();
                this.syncActiveKeywordsToBackend();
                this.showToast("Auto keywords disabled.", "info");
            }
              },


        restoreAutoDomainKeywords() {
            if (this.lastAutoDomainKeywordsList.length) {
                console.log("[AutoKeywords] Restoring last list");
                this.autoDomainKeywordsList = [...this.lastAutoDomainKeywordsList];
                this._updateActiveSet();
                this.syncActiveKeywordsToBackend();
                this.showToast("Restored last auto keywords set.", "success");
            } else {
                console.log("[AutoKeywords] No last list; fetching from backend");
                this.fetchAutoDomainKeywords().then(() => {
                    if (this.allAutoDomainKeywordsList.length > 0) {
                        this.showToast("Fetched keywords from backend.", "success");
                    } else {
                        this.showToast("No previous set to restore and none on backend.", "info");
                    }
                    console.log("Current list:", this.autoDomainKeywordsList);
                    console.log("Last saved list:", this.lastAutoDomainKeywordsList);
                });
                }
            },

            _updateActiveSet() {
                this.activeAutoDomainKeywordsSet = new Set(this.autoDomainKeywordsList);
            },

            // === Backend Sync ===
            async syncActiveKeywordsToBackend() {
                try {
                    console.log("Syncing keywords to backend:", Array.from(this.activeAutoDomainKeywordsSet));
                    const response = await fetch('/update_auto_domain_keywords', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            keywords: Array.from(this.activeAutoDomainKeywordsSet)
                        })
                    });
                    const result = await response.json();
                    if (!result.success) {
                        console.error("Backend sync failed:", result.error);
                        throw new Error(result.error || "Unknown error");
                    }
                    console.log("Keywords synced successfully");
                } catch (e) {
                    console.error("Sync error:", e);
                    this.showToast(`Failed to update keywords: ${e.message}`, "error");
                }
            },


            // === UI Feedback ===
            showToast(message, type = 'info', duration = 3000) {
                this.toast.message = message;
                this.toast.type = type;
                this.toast.show = true;
                clearTimeout(this.toast.timeout);
                this.toast.timeout = setTimeout(() => this.toast.show = false, duration);
            },
        

    // === UTILITY METHODS ===
    isLoading() {
        const msg = this.statusMessage.toLowerCase();
        return msg.includes('loading') || msg.includes('saving') ||
            msg.includes('creating') || msg.includes('removing') ||
            msg.includes('activating') || msg.includes('ingesting') ||
            msg.includes('uploading') || msg.includes('sending') ||
            msg.includes('replying') || msg.includes('applying');
    },


    isLoadingChat() {
        return this.isStreaming ||
            this.statusMessage === 'Sending...' ||
            this.statusMessage === 'Assistant is replying...';
    },

    formatTimestamp(isoString) {
        if (!isoString) return '';
        try {
            return new Date(isoString).toLocaleString();
        } catch (e) {
            return isoString;
        }
    },

    scrollToBottom() {
        this.$nextTick(() => {
            const el = this.$refs.chatHistoryContainer;
            if (el) el.scrollTop = el.scrollHeight;
        });
    },

            data() {
                return {
                    isSavingTopN: false,
                    // …other state…
                }
            },
            
            methods: {
                async applyTopNKeywords() {
                    if (this.isSavingTopN) return;        // drop any extra clicks
                    this.isSavingTopN = true;
                    try {
                        await this.saveTopNToConfig();      // your existing save method
                        this.showToast("Top-N saved", "success");
                    } catch (e) {
                        this.showToast(`Failed to save Top-N: ${e}`, "error");
                    } finally {
                        this.isSavingTopN = false;
                    }
                },
                // …
            },        

        async saveConfig() {
            // Prevent double‐submit
            if (this.isLoading()) {
                console.warn('[Button Click] Save Config ignored, already loading:', this.statusMessage);
                if (this.showToast) this.showToast("Please wait for the current operation to finish.", "info");
                return false;
            }

            console.log("Saving configuration via /save_config...");
            this.statusMessage = 'Saving Config...';

            // ── 1) Sync doc‐freq mode & compute the one true min_doc_freq ──
            const dke = this.formConfig.domain_keyword_extraction;
            // Persist the chosen mode
            dke.min_doc_freq_mode = this.docFreqMode;

            // Compute canonical min_doc_freq
            if (this.docFreqMode === 'absolute') {
                // Use the absolute input (or fall back to existing min_doc_freq)
                dke.min_doc_freq = dke.min_doc_freq_abs ?? dke.min_doc_freq;
            } else {
                // Fraction mode → compute count from totalDocs
                dke.min_doc_freq = (this.totalDocs && dke.min_doc_freq_frac != null)
                    ? Math.ceil(dke.min_doc_freq_frac * this.totalDocs)
                    : dke.min_doc_freq;
            }

            // (Optional) Remove UI‐only helpers so backend only sees the one field
            //delete dke.min_doc_freq_abs;
            //delete dke.min_doc_freq_frac;

            try {
                // ── 2) POST to backend ──
                const response = await fetch('/save_config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.formConfig)
                });
                const result = await response.json();

                // 3) Error handling
                if (!response.ok || !result.success) {
                    const errorMsg = result.error || result.message || `HTTP error ${response.status}`;
                    throw new Error(errorMsg);
                }

                // 4) Merge back any validated defaults from server
                if (result.config) {
                    console.log("[saveConfig] Merging validated config from backend");
                    this.deepMerge(this.formConfig, result.config);
                }

                // 5) Notify & mark clean
                this.showToast(result.message || 'Configuration saved successfully!', 'success');
                this._markConfigClean();
                return true;

            } catch (error) {
                console.error('Configuration Save Error:', error);
                this.statusMessage = 'Save Error';
                this.showToast(`Error saving config: ${error.message}`, 'error');
                return false;

            } finally {
                // Reset status
                if (this.statusMessage === 'Saving Config...') {
                    this.statusMessage = 'Idle';
                } else if (this.statusMessage === 'Save Error') {
                    setTimeout(() => {
                        if (this.statusMessage === 'Save Error') this.statusMessage = 'Idle';
                    }, 2000);
                }
            }
        },
              


                    // === INITIALIZATION METHOD (Called manually below) ===
            async loadInitialData() {
                console.log("[Init Method] loadInitialData() called.");
                this.statusMessage = 'Loading UI data...';
                try {
                    console.log("[Init Method] Starting API calls in Promise.all...");
                    await Promise.all([
                        (async () => {
                            console.log("[Init API] Starting loadInitialConfig...");
                            try {
                                await this.loadInitialConfig();
                                console.log("[Init API] Finished loadInitialConfig.");
                            } catch (e) {
                                console.error("[Init API] loadInitialConfig FAILED:", e);
                                throw e;
                            }
                        })(),
                        (async () => {
                            console.log("[Init API] Fetching document count...");
                            try {
                                const response = await fetch('/get-doc-count');
                                const data = await response.json();
                                this.totalDocs = data.total_docs;
                                console.log("[Init API] Document count loaded:", this.totalDocs);
                            } catch (e) {
                                console.warn("[Init API] Could not fetch doc count", e);
                            }
                        })(),
                        (async () => {
                            console.log("[Init API] Starting checkApiKeys...");
                            try {
                                await this.checkApiKeys();
                                console.log("[Init API] Finished checkApiKeys.");
                            } catch (e) {
                                console.error("[Init API] checkApiKeys FAILED:", e);
                            }
                        })(),
                        (async () => {
                            console.log("[Init API] Starting fetchPresets...");
                            try {
                                await this.fetchPresets();
                                console.log("[Init API] Finished fetchPresets.");
                            } catch (e) {
                                console.error("[Init API] fetchPresets FAILED:", e);
                            }
                        })(),
                        (async () => {
                            console.log("[Init API] Starting fetchWeaviateInstances...");
                            try {
                                await this.fetchWeaviateInstances();
                                console.log("[Init API] Finished fetchWeaviateInstances.");
                            } catch (e) {
                                console.error("[Init API] fetchWeaviateInstances FAILED:", e);
                            }
                        })(),
                        (async () => {
                            console.log("[Init API] Starting fetchSavedChats...");
                            try {
                                await this.fetchSavedChats();
                                console.log("[Init API] Finished fetchSavedChats.");
                            } catch (e) {
                                console.error("[Init API] fetchSavedChats FAILED:", e);
                            }
                        })(),
                        (async () => {
                            console.log("[Init API] Starting fetchAutoDomainKeywords...");
                            try {
                                await this.fetchAutoDomainKeywords();
                                console.log("[Init API] Finished fetchAutoDomainKeywords.");
                            } catch (e) {
                                console.error("[Init API] fetchAutoDomainKeywords FAILED:", e);
                            }
                        })(),
                    ]);

                    console.log("[Init Method] Promise.all finished.");

                    if (this.chatHistory.length === 0) {
                        console.log("[Init Method] Adding welcome message.");
                        this.chatHistory.push({
                            role: 'assistant',
                            text: 'Hello! Ask me anything.',
                            timestamp: new Date().toISOString()
                        });
                    } else {
                        console.log("[Init Method] Skipping welcome message (history already present).");
                    }

                    this.statusMessage = 'Idle';
                    console.log("[Init Method] init() finished successfully.");
                    this.scrollToBottom();
                    this.$nextTick(() => {
                        console.log("[Init Method] Focusing input area.");
                        this.$refs.inputArea?.focus();
                    });

                } catch (error) {
                    console.error("[Init Method] CRITICAL Error during init:", error);
                    this.statusMessage = 'Initialization Error!';
                    this.showToast(`Initialization failed: ${error.message}. Check console & backend.`, 'error', 10000);
                }
            },



                    // ADD THIS METHOD: Called by the button click in index.html
                    async runKeywordBuilderWithCheck() {
                        console.log(
                            '[Button Click] Extract Domain Keywords clicked. Checking for unsaved changes...');
                        // Use the confirmation helper to check for unsaved changes before proceeding
                        // It will call _performRunKeywordBuilder only if config is clean or after successful save
                        this._confirmUnsavedChanges('Extract Domain Keywords', this._performRunKeywordBuilder
                            .bind(this));
                    },

                    // ADD THIS METHOD: Contains the original keyword builder logic
                    async _performRunKeywordBuilder() {
                        // Check loading state *inside* the core logic as well
                        if (this.isLoading()) {
                            console.warn('[Button Click - Core] Keyword Builder ignored, already loading:', this
                                .statusMessage);
                            return;
                        }

                        console.log("[Keyword Builder] Running core logic...");
                        this.statusMessage = 'Extracting keywords...'; // Set loading status

                        // Get references to UI elements (consider using x-ref in HTML for cleaner Alpine integration later)
                        const keywordResultsDiv = document.getElementById('keywordResults');
                        const keywordListDiv = document.getElementById('keywordList');
                        const formElement = document.getElementById(
                            'keywordBuilderForm'); // Assuming form has this ID

                        if (!formElement || !keywordResultsDiv || !keywordListDiv) {
                            console.error(
                                "[Keyword Builder] Required DOM elements (form, results area) not found.");
                            this.statusMessage = 'UI Error';
                            if (this.showToast) this.showToast("Keyword builder UI elements missing.", "error");
                            setTimeout(() => {
                                if (this.statusMessage === 'UI Error') this.statusMessage = 'Idle';
                            }, 3000);
                            return;
                        }

                        // Hide previous results
                        keywordResultsDiv.style.display = 'none';
                        keywordListDiv.innerHTML = ''; // Clear previous list

                        // Prepare data from the form
                        const formData = new FormData(formElement);
                        const jsonData = {};
                        try {
                            for (const [key, value] of formData.entries()) {
                                if (key === 'no_pos_filter') {
                                    jsonData[key] = true; // Checkbox value
                                }
                                else if (key === 'diversity') {
                                    jsonData[key] = parseFloat(value);
                                }
                                else if (key === 'min_doc_freq_abs') {
                                    jsonData.min_doc_freq_abs = parseInt(value, 10) || null;
                                }
                                else if (key === 'min_doc_freq_frac') {
                                    jsonData.min_doc_freq_frac = parseFloat(value) || null;
                                    // Ensure conversion to number, handle potential NaN
                                    const numValue = parseInt(value, 10);
                                    jsonData[key] = isNaN(numValue) ? null :
                                        numValue; // Send null if not a number? Or default?
                                    if (isNaN(numValue)) console.warn(
                                        `[Keyword Builder] Invalid number for ${key}: ${value}`);
                                }
                                else {
                                    jsonData[key] = value;
                                }
                            }
                            // Manually add Weaviate port from config if needed by script
                            // jsonData['weaviate_http_port'] = this.formConfig.retrieval.WEAVIATE_HTTP_PORT;
                            // jsonData['collection'] = this.formConfig.retrieval.COLLECTION_NAME;
                            // Note: The backend route seems to get these from cfg now, so maybe not needed here. Verify backend logic.
                        }
                        catch (formError) {
                            console.error('[Keyword Builder] Error processing form data:', formError);
                            this.statusMessage = 'Form Error';
                            if (this.showToast) this.showToast("Error reading keyword settings.", "error");
                            setTimeout(() => {
                                if (this.statusMessage === 'Form Error') this.statusMessage = 'Idle';
                            }, 3000);
                            return;
                        }


                        console.log("[API Call] Sending request to /run_keyword_builder with data:", jsonData);

                        try {
                            // Call the backend endpoint
                            const response = await fetch('/run_keyword_builder',
                                {
                                    method: 'POST',
                                    headers:
                                    {
                                        'Content-Type': 'application/json'
                                    },
                                    body: JSON.stringify(jsonData)
                                });

                            const data = await response.json(); // Always try to parse

                            // Check for HTTP errors or backend reported failure
                            if (!response.ok || !data.success) {
                                // Extract detailed error message
                                let errorDetail = data.error || `HTTP error! Status: ${response.status}`;
                                if (data.details) errorDetail += ` Details: ${data.details}`;
                                if (data.full_error) console.error("Full Keyword Builder Error:\n", data
                                    .full_error);
                                throw new Error(errorDetail); // Throw to be caught by catch block
                            }

                            // --- Success Path ---
                            console.log("[API Resp] Keyword builder success:", data);
                            keywordResultsDiv.style.display = 'block'; // Show results area

                            if (data.keywords && data.keywords.length > 0) {
                                // Format and display keywords
                                const keywordsHtml = data.keywords.map(kw =>
                                    `<li class="text-xs">${kw.term}: ${kw.score.toFixed(4)}</li>` // Use list items
                                ).join('');
                                keywordListDiv.innerHTML =
                                    `<ul class="list-disc list-inside">${keywordsHtml}</ul>`; // Wrap in UL

                                // Add "Update Config" button dynamically
                                const updateConfigBtn = document.createElement('button');
                                updateConfigBtn.className =
                                    'btn btn-success mt-3 text-xs py-1 px-2'; // Use btn classes
                                updateConfigBtn.textContent = 'Update Config with These Keywords';
                                // IMPORTANT: Ensure updateConfigWithKeywords is accessible globally or via Alpine component instance
                                // If global:
                                
                                // If inside Alpine component (preferred):
                                // updateConfigBtn.onclick = () => Alpine.$data(document.getElementById('rootAppComponent')).updateConfigWithKeywords(data.keywords.map(kw => kw.term));
                                // call the Alpine component method
                                updateConfigBtn.onclick = () => this.updateConfigWithKeywords( data.keywords.map(kw => kw.term)   );
                                keywordListDiv.appendChild(updateConfigBtn);

                                if (this.showToast) this.showToast(`Extracted ${data.keywords.length} keywords.`,
                                    "success");
                                this.statusMessage = 'Keywords Extracted'; // Set completion status

                            }
                            else {
                                // Success response but no keywords found
                                keywordListDiv.innerHTML =
                                    `<p class="text-xs text-slate-600">${data.message || 'No keywords extracted based on current settings.'}</p>`;
                                if (this.showToast) this.showToast(data.message || 'No keywords extracted.',
                                    'info');
                                this.statusMessage = 'No Keywords Found'; // Set completion status
                            }

                        }
                        catch (error) {
                            // --- Error Path ---
                            console.error('[API Error] Run Keyword Builder FAILED:', error);
                            this.statusMessage = 'Keyword Error'; // Set error status
                            // Display error in the results area
                            keywordListDiv.innerHTML =
                                `<p class="text-xs text-red-600">Error: ${error.message}</p>`;
                            keywordResultsDiv.style.display = 'block';
                            // Show error toast
                            if (this.showToast) this.showToast(`Keyword extraction failed: ${error.message}`,
                                'error', 10000);

                        }
                        finally {
                            // --- Finally Block ---
                            // Reset status after a delay, unless another operation started
                            setTimeout(() => {
                                // Check against all possible completion/error statuses for this action
                                const relevantStatuses = ['Extracting keywords...', 'Keywords Extracted',
                                    'No Keywords Found', 'Keyword Error', 'Form Error', 'UI Error'];
                                if (relevantStatuses.includes(this.statusMessage)) {
                                    this.statusMessage = 'Idle';
                                }
                            }, 4000); // 4-second delay
                            // --- END Finally Block ---
                        }
                    }, // End of _performRunKeywordBuilder

                    configIsDirty() {
                        if (!this.savedConfig) {
                            // If savedConfig hasn't been loaded yet, assume not dirty (or handle as needed)
                            console.log("[Dirty Check] savedConfig not ready, assuming clean.");
                            return false;
                        }
                        try {
                            const currentString = JSON.stringify(this.formConfig);
                            const savedString = JSON.stringify(this.savedConfig);
                            const isDirty = currentString !== savedString;
                            if (isDirty) console.log("[Dirty Check] Config IS dirty.");
                            // else console.log("[Dirty Check] Config is clean."); // Optional: uncomment for debugging
                            return isDirty || this
                                .configDirtyExplicitlySet; // Combine deep compare with explicit flag
                        }
                        catch (e) {
                            console.error("[Dirty Check] Error comparing config states:", e);
                            return true; // Assume dirty if comparison fails
                        }
                    },



                    // Helper to reset dirty state (called after successful save/load)
                    _markConfigClean() {
                        try {
                            this.savedConfig = JSON.parse(JSON.stringify(this.formConfig)); // Deep copy current state
                            this.configDirtyExplicitlySet = false; // Reset explicit flag
                            console.log("[Dirty State] Marked config as clean.");
                        } catch (e) {
                            console.error("[Dirty State] Error marking config clean:", e);
                            // savedConfig might be invalid now, maybe reload?
                            this.savedConfig = null;
                        }
                    },

                    // Helper for actions requiring config save confirmation
            _confirmUnsavedChanges(actionName, actionFunction) {
                console.log(`[Confirm Action Check] Action: ${actionName}, Config Dirty: ${this.configIsDirty()}`);

                if (this.configIsDirty()) {
                    this.requireConfirmation({
                        title: 'Unsaved Configuration',
                        message: `You have unsaved configuration changes. Save them before starting '${actionName}'?`,
                        confirmButtonClass: 'bg-green-600 hover:bg-green-700', // Green for Save & Proceed
                        onConfirm: async () => {
                            console.log(`[Confirm Action] User chose Save & Proceed for ${actionName}.`);

                            try {
                                // 🚫 DO NOT set statusMessage manually here — let saveConfig handle it.
                                const saveSuccess = await this.saveConfig(); // Save config and wait

                                if (saveSuccess) {
                                    console.log(`[Confirm Action] Config saved. Proceeding with ${actionName}...`);
                                    actionFunction(); // Proceed with intended action
                                } else {
                                    console.warn(`[Confirm Action] Config save failed. Aborting ${actionName}.`);
                                    this.statusMessage = 'Save Failed';
                                    setTimeout(() => {
                                        if (this.statusMessage === 'Save Failed') this.statusMessage = 'Idle';
                                    }, 2000);
                                }
                            } catch (error) {
                                console.error(`[Confirm Action] Error during save-before-action for ${actionName}:`, error);
                                this.statusMessage = 'Save Error';
                                setTimeout(() => {
                                    if (this.statusMessage === 'Save Error') this.statusMessage = 'Idle';
                                }, 2000);
                            }
                        },
                        onCancel: () => {
                            console.log(`[Confirm Action] User cancelled ${actionName} due to unsaved changes.`);
                            if (this.showToast) this.showToast(`Action '${actionName}' cancelled.`, 'info');

                            // Reset status if left hanging
                            if (this.isLoading() && !this.isLoadingChat()) this.statusMessage = 'Idle';
                        }
                    });
                } else {
                    console.log(`[Confirm Action Check] Config clean. Proceeding directly with ${actionName}.`);
                    actionFunction();
                }
            },

                    adjustTextareaHeight(el) {
                        if (!el) el = this.$refs.inputArea;
                        if (!el) return;
                        const maxH = 150;
                        el.style.height = 'auto';
                        el.style.height = `${Math.min(el.scrollHeight, maxH)}px`;
                    },

            safeRenderMarkdown(text) {
                if (!text) return '';
                // Fallback: if libs aren’t available, HTML-escape the text
                if (!window.marked || !window.DOMPurify) {
                    return text
                        .replace(/&/g, '&amp;')
                        .replace(/</g, '&lt;')
                        .replace(/>/g, '&gt;')
                        .replace(/\n/g, '<br>');
                }
                // 1) Full Markdown → HTML
                const html = window.marked.parse(text);
                // 2) Sanitize HTML
                return DOMPurify.sanitize(html);
            },

                    showToast(message, type = 'info', duration = 3000) {
                        console.log(`[Toast (${type})] ${message}`);
                        this.toast.message = message;
                        this.toast.type = type;
                        this.toast.show = true;
                        clearTimeout(this.toast.timeout);
                        this.toast.timeout = setTimeout(() => {
                            this.toast.show = false;
                        }, duration);
                    },
                    requireConfirmation(options) {
                        console.log(
                            `[Confirm Req] Title: ${options.title || 'Confirmation'}, Message: ${options.message || 'Are you sure?'}`
                        );
                        // Ensure modal state exists (should be defined in initial state)
                        if (!this.confirmationModal || typeof this.confirmationModal !== 'object') {
                            console.error(
                                "[requireConfirmation] ERROR: confirmationModal state object is missing!");
                            // Optionally show a generic toast if available
                            if (this.showToast) this.showToast("Cannot display confirmation dialog.", "error");
                            return; // Prevent further execution
                        }
                        // Assign properties
                        this.confirmationModal.title = options.title || 'Confirmation';
                        this.confirmationModal.message = options.message || 'Are you sure?';
                        // Store the functions to be called later, ensuring they are functions
                        this.confirmationModal.onConfirm = (typeof options.onConfirm === 'function') ? options
                            .onConfirm : () => {
                            console.warn("No valid onConfirm action provided for modal.")
                        };
                        this.confirmationModal.onCancel = (typeof options.onCancel === 'function') ? options
                            .onCancel : () => {
                            /* Default no-op is fine */
                        };
                        this.confirmationModal.confirmButtonClass = options.confirmButtonClass ||
                            'bg-blue-600 hover:bg-blue-700'; // Default style
                        // Show the modal
                        this.confirmationModal.show = true;
                    },

                    confirmAction() {
                        console.log("[Modal Action] Confirm button clicked.");
                        // Check if modal state and callback exist
                        if (!this.confirmationModal || typeof this.confirmationModal.onConfirm !== 'function') {
                            console.error(
                                "[confirmAction] ERROR: Cannot execute confirmation, modal state or onConfirm callback is invalid."
                            );
                            if (this.showToast) this.showToast("Confirmation action failed (internal error).",
                                "error");
                            // Still hide the modal if possible
                            if (this.confirmationModal) this.confirmationModal.show = false;
                            return;
                        }
                        try {
                            // Execute the callback function stored when requireConfirmation was called
                            this.confirmationModal.onConfirm(); // Call the stored confirm logic
                        }
                        catch (error) {
                            console.error("Error executing confirmation action callback:", error);
                            // Use showToast if it exists
                            if (this.showToast) this.showToast(
                                `Operation failed during confirmation: ${error.message}`, 'error');
                        }
                        finally {
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
                        }
                        catch (error) {
                            console.error("Error executing cancel action callback:", error);
                            // Rarely need user notification for cancel error
                        }
                        finally {
                            // Always hide the modal
                            this.confirmationModal.show = false;
                            // Optional: Reset callbacks
                            // this.confirmationModal.onConfirm = () => {};
                            // this.confirmationModal.onCancel = () => {};
                        }
                    },
                    handleChatInputKeydown(event) {
                        if (event.key === 'Enter' && !event.shiftKey) {
                            event.preventDefault();
                            this.sendMessage();
                        }
                        this.$nextTick(() => this.adjustTextareaHeight(event.target));
                    },
                    handleFileSelect(event) {
                        this.filesToUpload = Array.from(event.target.files || []);
                        this.selectedFileNames = this.filesToUpload.map(f => f.name);
                    },


                    // Helper: is a keyword active?
                    isKeywordActive(kw) {
                        return this.activeAutoDomainKeywordsSet.has(kw);
                    },

                    // Toggle keyword active/inactive
                    toggleKeyword(kw) {
                        if (this.activeAutoDomainKeywordsSet.has(kw)) {
                            this.activeAutoDomainKeywordsSet.delete(kw);
                        } else {
                            this.activeAutoDomainKeywordsSet.add(kw);
                        }
                        this.syncActiveKeywordsToBackend();
                    },


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
                        }
                        catch (error) {
                            console.error("[API Error] checkApiKeys FAILED:", error);
                            // Keep defaults (all false) on error
                            this.apiKeyStatus = {
                                deepseek: false,
                                openai: false,
                                anthropic: false,
                                cohere: false
                            };
                            if (this.showToast) this.showToast('Could not check API key status.', 'error');
                        }
                    },

                    // --- Weaviate Instance Management ---
                    async fetchWeaviateInstances() {
                        console.log("[API Call] Fetching Weaviate instances from /list_weaviate_instances...");
                        this.statusMessage = 'Loading Weaviate instances...'; // Show loading status
                        try {
                            const response = await fetch(
                                '/list_weaviate_instances'); // Ensure Flask has this GET endpoint
                            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                            const data = await response.json();
                            console.log("[API Resp] Received Weaviate data:", data);
                            if (!Array.isArray(data)) throw new Error(
                                "Received invalid data format for instances.");

                            // Use the received list directly (active flag is set by backend)
                            this.weaviateInstances = data;

                            console.log("[API Logic] Processed Weaviate instances:", this.weaviateInstances);
                            // Only set status to Idle if this specific fetch succeeded
                            if (this.statusMessage === 'Loading Weaviate instances...') {
                                this.statusMessage = 'Idle';
                            }
                        }
                        catch (error) {
                            console.error("[API Error] fetchWeaviateInstances FAILED:", error);
                            if (this.showToast) this.showToast(
                                `Error loading Weaviate instances: ${error.message}`, 'error');
                            this.weaviateInstances = []; // Clear list on error
                            this.statusMessage = 'Error loading instances'; // Keep error status
                        }
                    },

                    async createWeaviateInstance() {
                        // Added console log from previous step for debugging click
                        console.log('[Button Click] Create button clicked. isLoading:', this.isLoading(),
                            'Instance Name:', this.newInstanceName);
                        const instanceName = this.newInstanceName.trim();
                        if (!instanceName) {
                            if (this.showToast) this.showToast("Please enter a name for the new instance.",
                                "info");
                            return;
                        }
                        if (this.isLoading()) {
                            console.warn('[Button Click] Create ignored, already loading:', this.statusMessage);
                            return; // Prevent action if already busy
                        }

                        console.log(`[API Call] Creating Weaviate instance: ${instanceName}`);
                        this.statusMessage = `Creating ${instanceName}...`;
                        try {
                            const response = await fetch('/create_weaviate_instance',
                                { // Ensure Flask POST endpoint exists
                                    method: 'POST',
                                    headers:
                                    {
                                        'Content-Type': 'application/json'
                                    },
                                    body: JSON.stringify(
                                        {
                                            instance_name: instanceName
                                        })
                                });
                            const result = await response.json();
                            if (!response.ok) throw new Error(result.error ||
                                `HTTP error! Status: ${response.status}`);

                            if (this.showToast) this.showToast(result.message ||
                                `Instance '${instanceName}' creating...`, 'success');
                            this.newInstanceName = ''; // Clear input
                            await this.fetchWeaviateInstances(); // Refresh list after creation attempt

                        }
                        catch (error) {
                            console.error('[API Error] Create Instance FAILED:', error);
                            this.statusMessage = 'Create Error';
                            if (this.showToast) this.showToast(`Error creating instance: ${error.message}`,
                                'error');
                        }
                        finally {
                            // Reset status only if it wasn't changed by another operation in the meantime
                            if (this.statusMessage === `Creating ${instanceName}...`) {
                                setTimeout(() => {
                                    this.statusMessage = 'Idle';
                                }, 1000);
                            }
                        }
                    },

            
                    // Uses confirmation modal
                    removeWeaviateInstanceWithConfirmation(instanceName) {
                        if (!instanceName || instanceName === "Default (from config)") {
                            if (this.showToast) this.showToast("Cannot remove the default configuration entry.",
                                "info");
                            return;
                        }
                        if (this.isLoading()) {
                            console.warn('[Button Click] Remove ignored, already loading:', this.statusMessage);
                            return; // Prevent action if already busy
                        }

                        this.requireConfirmation(
                            {
                                title: 'Remove Weaviate Instance',
                                message: `Are you sure you want to remove instance "${instanceName}"? This will stop and delete its container and data volume.`,
                                confirmButtonClass: 'bg-red-600 hover:bg-red-700', // Red confirm button
                                onConfirm: async () => { // The actual async logic
                                    console.log(`[Confirm Action] Removing Weaviate instance: ${instanceName}`);
                                    this.statusMessage = `Removing ${instanceName}...`;
                                    try {
                                        const response = await fetch('/remove_weaviate_instance',
                                            { // Ensure Flask POST endpoint exists
                                                method: 'POST', // Or DELETE if app.py uses DELETE
                                                headers:
                                                {
                                                    'Content-Type': 'application/json'
                                                },
                                                body: JSON.stringify(
                                                    {
                                                        instance_name: instanceName
                                                    })
                                            });
                                        const result = await response.json();
                                        if (!response.ok) throw new Error(result.error ||
                                            `HTTP error ${response.status}`);

                                        if (this.showToast) this.showToast(result.message ||
                                            `Instance '${instanceName}' removed.`, 'success');
                                        // Refresh instances and potentially config after removal
                                        await this.fetchWeaviateInstances();
                                        await this
                                            .loadInitialConfig(); // Re-fetch config in case active instance changed

                                    }
                                    catch (error) {
                                        console.error('[API Error] Remove Instance FAILED:', error);
                                        this.statusMessage = 'Remove Error';
                                        if (this.showToast) this.showToast(
                                            `Error removing instance: ${error.message}`, 'error');
                                    }
                                    finally {
                                        if (this.statusMessage === `Removing ${instanceName}...`) {
                                            setTimeout(() => {
                                                this.statusMessage = 'Idle';
                                            }, 1000);
                                        }
                                    }
                                },
                                onCancel: () => {
                                    console.log(`[Confirm Action] Cancelled removal for: ${instanceName}`);
                                }
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
                    const response = await fetch('/select_weaviate_instance', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ instance_name: instanceName })
                    });
                    const result = await response.json();
                    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);

                    // show success toast
                    if (this.showToast) {
                        this.showToast(
                            result.message || `Instance '${instanceName}' activated.`,
                            'success'
                        );
                    }

                    // reload core config (so your input bindings, endpoint, etc. update)
                    await this.loadInitialConfig();

                    // **in‐place** flip the active flag locally
                    this.weaviateInstances.forEach(inst => {
                        inst.active = (inst.name === instanceName);
                    });

                    this.statusMessage = 'Idle';
                }
                catch (error) {
                    console.error('[API Error] Activate Instance FAILED:', error);
                    this.statusMessage = 'Activation Error';
                    if (this.showToast) {
                        this.showToast(
                            `Error activating instance: ${error.message}`,
                            'error'
                        );
                    }
                    setTimeout(() => {
                        if (
                            this.statusMessage === `Activating ${instanceName}...` ||
                            this.statusMessage === 'Activation Error'
                        ) {
                            this.statusMessage = 'Idle';
                        }
                    }, 3000);
                }
            },


                    // --- Chat Functionality ---

                    async sendMessage() {
                        // Use the confirmation helper to check for unsaved changes before proceeding
                        // It will call _performSendMessage only if config is clean or after successful save
                        this._confirmUnsavedChanges('Send Message', this._performSendMessage.bind(this));
                    },

                    async _performSendMessage() { // Contains the original message sending logic
                        const queryToSend = this.userInput.trim();
                        if (!queryToSend) {
                            // Use showToast if available (assuming it's defined elsewhere in your component)
                            if (this.showToast) this.showToast("Please enter a message.", "info", 2000);
                            return; // Don't proceed if input is empty
                        }

                        // Prevent sending if already processing a chat response
                        if (this.isLoadingChat()) {
                            console.warn('[Chat Action] Send ignored, already processing:', this.statusMessage);
                            // Optionally notify the user
                            if (this.showToast) this.showToast("Please wait for the current response.", "info");
                            return;
                        }

                        // --- Proceed with sending ---
                        // Add user message immediately to the UI
                        this.chatHistory.push(
                            {
                                role: 'user',
                                text: queryToSend,
                                timestamp: new Date().toISOString()
                            });

                        this.userInput = ''; // Clear input field
                        this.adjustTextareaHeight(); // Adjust height after clearing
                        this.scrollToBottom(); // Scroll down to show the new message

                        // Set loading/streaming state
                        this.isStreaming = true;
                        this.statusMessage = 'Assistant is replying...';

                        console.log(
                            `[API Call] Sending query to /run_pipeline: ${queryToSend.substring(0, 50)}...`);

                        try {
                            // Call the backend pipeline endpoint
                            const response = await fetch('/run_pipeline',
                                { // Ensure Flask POST endpoint accepts JSON
                                    method: 'POST',
                                    headers:
                                    {
                                        'Content-Type': 'application/json'
                                    },
                                    body: JSON.stringify(
                                        {
                                            query: queryToSend
                                        }) // Send query in JSON body
                                });

                            // Check for HTTP errors
                            if (!response.ok) {
                                let errorText = `Error: ${response.statusText} (Code: ${response.status})`;
                                try {
                                    const errorJson = await response.json();
                                    if (typeof errorJson.text === 'string') {
                                        errorText = errorJson.text;
                                    } else if (typeof errorJson.error === 'string') {
                                        errorText = errorJson.error;
                                    } else {
                                        errorText = `Unexpected error response: ${JSON.stringify(errorJson)}`;
                                    }
                                } catch (e) {
                                    console.warn("Could not parse error response as JSON:", e);
                                }
                                throw new Error(errorText);
                            }

                            // Parse successful JSON response
                            const result = await response
                                .json(); // Expect { role: 'assistant', text: '...', sources: [...], error: false, timestamp: '...' }
                            console.log("[API Resp] Received from /run_pipeline:", result);

                            // Add assistant response (or error reported *by* the backend) to chat history
                            this.chatHistory.push(
                                {
                                    role: 'assistant',
                                    text: result.text || result.response ||
                                        "[No response content]", // Handle potential key variations
                                    sources: result.sources, // Attach sources if backend provides them
                                    timestamp: result.timestamp || new Date()
                                        .toISOString(), // Use backend timestamp if provided
                                    error: result.error || false // Include error flag from backend response
                                });

                        }
                        catch (error) {
                            // Handle fetch errors or errors thrown from !response.ok check
                            console.error('[API Error] Send Message FAILED:', error);

                            // Add a clear error message to the chat history for the user
                            this.chatHistory.push(
                                {
                                    role: 'assistant',
                                    text: `Sorry, an error occurred: ${error.stack}`,
                                    timestamp: new Date().toISOString(),
                                    error: true // Mark this message as an error
                                });

                            // Optionally show a toast notification for the error
                            if (this.showToast) this.showToast(`Error: ${error.message}`, 'error', 5000);

                        }
                        finally {
                            // Always executed after try/catch
                            this.isStreaming = false; // End streaming/loading state
                            this.statusMessage = 'Idle'; // Reset status
                            this.scrollToBottom(); // Scroll down to show the final response/error

                            // Re-focus the input area for the next message
                            // Use $nextTick to ensure DOM is updated before focusing
                            this.$nextTick(() => {
                                this.$refs.inputArea?.focus();
                            });
                        }
                    }, // End of _performSendMessage

                    // --- Saved Chats ---
                    async fetchSavedChats() {
                        console.log("[API Call] Fetching saved chats from /list_chats...");
                        try {
                            const response = await fetch('/list_chats'); // Ensure Flask GET endpoint exists
                            if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                            const loadedChats = await response.json();
                            this.savedChats = Array.isArray(loadedChats) ? loadedChats :
                                []; // Ensure it's an array
                            console.log("[API Resp] Saved chats loaded:", this.savedChats.length);
                        }
                        catch (error) {
                            console.error("[API Error] fetchSavedChats FAILED:", error);
                            this.savedChats = []; // Ensure array on error
                            if (this.showToast) this.showToast(`Error loading saved chat list: ${error.message}`,
                                'error');
                        }
                    },

                    async saveChat() {
                        if (this.isLoading() || !Array.isArray(this.chatHistory) || this.chatHistory.length ===
                            0) {
                            if (this.showToast) this.showToast("Nothing to save in current chat.", "info");
                            return;
                        }
                        // Use prompt for simplicity, could be replaced with a modal input
                        const chatName = prompt("Enter a name for this chat:",
                            `Chat ${new Date().toLocaleDateString()}`);
                        if (!chatName) return; // User cancelled

                        console.log(`[API Call] Saving current chat as: ${chatName}`);
                        this.statusMessage = 'Saving chat...';
                        try {
                            const response = await fetch('/save_chat',
                                { // Ensure Flask POST endpoint accepts JSON
                                    method: 'POST',
                                    headers:
                                    {
                                        'Content-Type': 'application/json'
                                    },
                                    // Send name and the current history state
                                    body: JSON.stringify(
                                        {
                                            name: chatName,
                                            history: this.chatHistory
                                        })
                                });
                            const result = await response.json();
                            if (!response.ok || !result.success) throw new Error(result.error ||
                                `HTTP error! Status: ${response.status}`);

                            if (this.showToast) this.showToast(result.message || `Chat '${chatName}' saved.`,
                                'success');
                            await this.fetchSavedChats(); // Refresh list

                        }
                        catch (error) {
                            console.error('[API Error] Save Chat FAILED:', error);
                            this.statusMessage = 'Save Error';
                            if (this.showToast) this.showToast(`Error saving chat: ${error.message}`, 'error');
                        }
                        finally {
                            if (this.statusMessage === 'Saving chat...') {
                                setTimeout(() => {
                                    this.statusMessage = 'Idle';
                                }, 1000);
                            }
                        }
                    },

                    // Uses confirmation modal
                    loadChatWithConfirmation(chatId) {
                        if (!chatId || this.isLoading()) return;
                        const chatToLoad = this.savedChats.find(c => c.id === chatId);
                        if (!chatToLoad) {
                            console.error("Chat ID not found in loaded list:", chatId);
                            return;
                        }

                        this.requireConfirmation(
                            {
                                title: 'Load Chat',
                                message: `Load chat "${chatToLoad.name || chatId}"? This will replace the current chat history.`,
                                onConfirm: async () => {
                                    console.log(`[Confirm Action] Loading chat: ${chatId}`);
                                    this.statusMessage = 'Loading chat...';
                                    try {
                                        const response = await fetch(
                                            `/load_chat/${chatId}`); // Ensure Flask GET endpoint exists
                                        if (!response.ok) throw new Error(
                                            `HTTP ${response.status}: ${response.statusText}`);
                                        const loadedChat = await response
                                            .json(); // Expect { id: ..., name: ..., history: [...] }

                                        if (!Array.isArray(loadedChat.history)) throw new Error(
                                            "Loaded chat history is not valid.");

                                        this.chatHistory = loadedChat.history; // Replace current history
                                        if (this.showToast) this.showToast(
                                            `Chat "${loadedChat.name || chatId}" loaded.`, 'success');
                                        this.scrollToBottom();

                                    }
                                    catch (error) {
                                        console.error('[API Error] Load Chat FAILED:', error);
                                        this.statusMessage = 'Load Error';
                                        if (this.showToast) this.showToast(`Error loading chat: ${error.message}`,
                                            'error');
                                    }
                                    finally {
                                        this.statusMessage = 'Idle';
                                    }
                                },
                                onCancel: () => {
                                    console.log(`[Confirm Action] Cancelled load chat: ${chatId}`);
                                }
                            });
                    },

                    // Uses confirmation modal
                    deleteChatWithConfirmation(chatId) {
                        if (!chatId || this.isLoading()) return;
                        const chatToDelete = this.savedChats.find(c => c.id === chatId);
                        if (!chatToDelete) {
                            console.error("Chat ID not found in loaded list:", chatId);
                            return;
                        }

                        this.requireConfirmation(
                            {
                                title: 'Delete Saved Chat',
                                message: `Are you sure you want to permanently delete the saved chat "${chatToDelete.name || chatId}"?`,
                                confirmButtonClass: 'bg-red-600 hover:bg-red-700',
                                onConfirm: async () => {
                                    console.log(`[Confirm Action] Deleting chat: ${chatId}`);
                                    this.statusMessage = 'Deleting chat...';
                                    try {
                                        const response = await fetch(`/delete_chat/${chatId}`,
                                            {
                                                method: 'DELETE'
                                            }); // Ensure Flask DELETE endpoint exists
                                        const result = await response.json();
                                        if (!response.ok || !result.success) throw new Error(result.error ||
                                            `HTTP error! Status: ${response.status}`);

                                        if (this.showToast) this.showToast(result.message || 'Chat deleted.',
                                            'success');
                                        await this.fetchSavedChats(); // Refresh list

                                    }
                                    catch (error) {
                                        console.error('[API Error] Delete Chat FAILED:', error);
                                        this.statusMessage = 'Delete Error';
                                        if (this.showToast) this.showToast(`Error deleting chat: ${error.message}`,
                                            'error');
                                    }
                                    finally {
                                        if (this.statusMessage === 'Deleting chat...') {
                                            setTimeout(() => {
                                                this.statusMessage = 'Idle';
                                            }, 1000);
                                        }
                                    }
                                },
                                onCancel: () => {
                                    console.log(`[Confirm Action] Cancelled delete chat: ${chatId}`);
                                }
                            });
                    },

                    // --- File Handling & Ingestion ---
                    async uploadFiles() {
                        // Added console log from previous step for debugging click
                        console.log('[Button Click] Upload files clicked. isLoading:', this.isLoading(),
                            'File Count:', this.filesToUpload.length);
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
                        this.filesToUpload.forEach(file => formData.append('files',
                            file)); // 'files' must match Flask request.files.getlist key

                        try {
                            const response = await fetch('/upload_files',
                                {
                                    method: 'POST',
                                    body: formData
                                }); // Ensure Flask POST endpoint exists
                            const result = await response.json();
                            if (!response.ok || !result.success) throw new Error(result.error ||
                                `HTTP error! Status: ${response.status}`);

                            if (this.showToast) this.showToast(
                                `Files uploaded: ${result.files?.join(', ') || 'OK'}`, 'success');
                            this.filesToUpload = []; // Clear selection state
                            this.selectedFileNames = [];
                            // Clear the actual file input element visually
                            const fileInput = document.getElementById(
                                'fileUpload'); // Use the correct ID from index.html
                            if (fileInput) fileInput.value = '';

                        }
                        catch (error) {
                            console.error('[API Error] Upload Files FAILED:', error);
                            this.statusMessage = 'Upload Error';
                            if (this.showToast) this.showToast(`Upload failed: ${error.message}`, 'error');
                        }
                        finally {
                            if (this.statusMessage === 'Uploading files...') {
                                setTimeout(() => {
                                    this.statusMessage = 'Idle';
                                }, 1500);
                            }
                        }
                    },

                    // --- File Handling & Ingestion ---

                    async startIngestion() {
                        console.log(
                            '[Button Click] Start Full Ingestion clicked. Checking for unsaved changes...');
                        this._confirmUnsavedChanges('Full Ingestion', this._performStartIngestion.bind(this));
                    },

                    async _performStartIngestion() { // Contains the original full ingestion logic
                        // Check loading state *inside* the core logic as well, in case user double-clicks
                        // or confirmation takes time.
                        if (this.isLoading()) {
                            console.warn('[Button Click - Core] Full Ingestion ignored, already loading:', this
                                .statusMessage);
                            // Optionally show toast, though it might be annoying if shown after confirmation
                            // if (this.showToast) this.showToast("Operation already in progress.", "info");
                            return;
                        }

                        console.log("[API Call] Starting full ingestion via /start_ingestion...");
                        this.statusMessage = 'Starting Full Ingestion...'; // Set loading status for UI feedback

                        try {
                            // Call the Flask backend endpoint for full ingestion
                            const response = await fetch('/start_ingestion',
                                {
                                    method: 'POST'
                                }); // Ensure Flask endpoint exists

                            // Always try to parse the response, even for errors, as backend might send details
                            const result = await response.json();

                            // Check for HTTP errors first
                            if (!response.ok) {
                                // Handle 503, 500, 404 etc. Use error message from backend if available.
                                let errorDetail = result.error || `HTTP error! Status: ${response.status}`;
                                // Log traceback if backend provided it (helpful for debugging server-side issues)
                                if (result.traceback) console.error("Ingestion Traceback:\n", result.traceback);
                                throw new Error(errorDetail); // Throw to be caught by the catch block
                            }

                            // --- Response Handling ---
                            if (result.status === 'centroid_missing') {
                                // Show popup for user to enter/select centroid path
                                let newPath = prompt(result.message + "\nEnter new centroid file path (or select):", this.formConfig.paths.DOMAIN_CENTROID_PATH);
                                if (newPath) {
                                    await this._createCentroidFile(newPath, this.formConfig.paths.DOCUMENT_DIR);
                                    // Optionally, retry ingestion after centroid creation:
                                    await this._performStartIngestion();
                                }
                                return; // Don't proceed further
                            }

                            // --- Success Path ---
                            let successMsg = result.message || 'Full ingestion finished.';
                            if (result.stats) {
                                const stats = result.stats;
                                const processed = stats.processed_chunks !== undefined ? stats.processed_chunks :
                                    stats.processed_files !== undefined ? stats.processed_files : 'N/A';
                                const inserted = stats.inserted !== undefined ? stats.inserted : 'N/A';
                                const errors = stats.errors !== undefined ? stats.errors : 0;
                                successMsg +=
                                    ` (Processed: ${processed}, Inserted: ${inserted}, Errors: ${errors})`;
                            }
                            if (this.showToast) this.showToast(successMsg, 'success', 8000);
                            this.statusMessage = 'Ingestion Complete';

                        }
                        catch (error) {
                            // --- Error Path ---
                            console.error('[API Error] Start Full Ingestion FAILED:', error);
                            this.statusMessage = 'Ingestion Error'; // Set error status
                            // Show a more detailed error toast for longer
                            if (this.showToast) this.showToast(`Ingestion failed: ${error.message}`, 'error',
                                10000);

                        }
                        finally {
                            // --- Finally Block ---
                            // Reset status after a delay to allow user to see 'Complete' or 'Error'
                            // Only reset if the status hasn't been changed by another concurrent operation.
                            setTimeout(() => {
                                if (this.statusMessage === 'Starting Full Ingestion...' || this
                                    .statusMessage === 'Ingestion Complete' || this.statusMessage ===
                                    'Ingestion Error') {
                                    this.statusMessage = 'Idle';
                                }
                            }, 4000); // 4-second delay before resetting status
                            // --- END Finally Block ---
                        }
                    }, // End of _performStartIngestion

            async _createCentroidFile(centroidPath, dataFolder) {
                try {
                    const response = await fetch('/create_centroid', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                        body: `centroid_path=${encodeURIComponent(centroidPath)}&data_folder=${encodeURIComponent(dataFolder)}`
                    });
                    const data = await response.json();
                    if (data.status === 'created') {
                        alert("Centroid file created successfully. Starting ingestion...");
                    } else {
                        alert("Failed to create centroid file: " + data.message);
                    }
                } catch (err) {
                    alert("AJAX error: " + err);
                }
            
                    if(data.status === 'created') {
                alert("Centroid file created successfully. Starting ingestion...");
                await this._performStartIngestion();
            }
            },

                    // --- File Handling & Ingestion ---

                    async startIncrementalIngestion() {
                        console.log(
                            '[Button Click] Start Incremental Ingestion clicked. Checking for unsaved changes...'
                        );
                        // Use the confirmation helper to check for unsaved changes before proceeding
                        // It will call _performStartIncrementalIngestion only if config is clean or after successful save
                        this._confirmUnsavedChanges('Incremental Ingestion', this
                            ._performStartIncrementalIngestion.bind(this));
                    },

            async _performStartIncrementalIngestion() {
                // Prevent double execution
                if (this.isLoading()) {
                    console.warn('[Button Click - Core] Incremental Ingestion ignored, already loading:', this.statusMessage);
                    return;
                }

                console.log("[API Call] Starting incremental ingestion via /ingest_block...");
                this.statusMessage = 'Starting Incremental Ingestion...';

                // Get config values (adapt to your model/Alpine structure)
                const dataFolder = this.formConfig.paths.DOCUMENT_DIR;
                const centroidPath = this.formConfig.paths.DOMAIN_CENTROID_PATH;
                const centroidUpdateMode = this.formConfig.centroidUpdateMode; // 'always', 'never', or 'auto'

                try {
                    // Send all necessary params to backend
                    // Build params safely, coerce undefined → default 0.05
                    const params = new URLSearchParams();
                    params.append('data_folder', dataFolder);
                    params.append('centroid_path', centroidPath);
                    params.append('centroid_update_mode', centroidUpdateMode);
                    // If formConfig.centroidAutoThreshold is missing or not a number, use 0.05
                    const rawThreshold = this.formConfig.centroidAutoThreshold;
                    const thresholdNum = typeof rawThreshold === 'number'
                        ? rawThreshold
                        : parseFloat(rawThreshold);
                    params.append(
                        'centroid_auto_threshold',
                        (Number.isFinite(thresholdNum) ? thresholdNum : 0.05).toString()
                    );

                    const response = await fetch('/ingest_block', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                        body: params.toString()
                    });

                    const result = await response.json();

                    // Handle centroid missing (same as full ingestion)
                    if (result.status === 'centroid_missing') {
                        let newPath = prompt(result.message + "\nEnter new centroid file path (or select):", centroidPath);
                        if (newPath) {
                            await this._createCentroidFile(newPath, dataFolder);
                            await this._performStartIncrementalIngestion();
                        }
                        return;
                    }

                    // Check for HTTP errors
                    if (!response.ok) {
                        let errorDetail = result.error || `HTTP error! Status: ${response.status}`;
                        if (result.traceback) console.error("Ingestion Traceback:\n", result.traceback);
                        throw new Error(errorDetail);
                    }

                    // Success path: show stats if available
                    let successMsg = result.message || 'Incremental ingestion finished.';
                    if (result.stats) {
                        const stats = result.stats;
                        const processed = stats.processed_files !== undefined ? stats.processed_files :
                            stats.processed_chunks !== undefined ? stats.processed_chunks : 'N/A';
                        const inserted = stats.inserted !== undefined ? stats.inserted : 'N/A';
                        const errors = stats.errors !== undefined ? stats.errors : 0;
                        successMsg += ` (Processed: ${processed}, Inserted: ${inserted}, Errors: ${errors})`;
                    }
                    if (this.showToast) this.showToast(successMsg, 'success', 6000);
                    this.statusMessage = 'Ingestion Complete';

                } catch (error) {
                    // Error path
                    console.error('[API Error] Start Incremental Ingestion FAILED:', error);
                    this.statusMessage = 'Ingestion Error';
                    if (this.showToast) this.showToast(`Incremental ingestion failed: ${error.message}`, 'error', 10000);

                } finally {
                    // Reset status after a delay
                    setTimeout(() => {
                        if (
                            this.statusMessage === 'Starting Incremental Ingestion...' ||
                            this.statusMessage === 'Ingestion Complete' ||
                            this.statusMessage === 'Ingestion Error'
                        ) {
                            this.statusMessage = 'Idle';
                        }
                    }, 4000);
                }
            },



            async loadInitialConfig() {
                console.log("Fetching initial config from /get_config...");
                try {
                    const response = await fetch('/get_config');
                    if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    const loadedConfig = await response.json();

                    // Helper for safe array conversion
                    const ensureArray = (value) => {
                        if (Array.isArray(value)) return value;
                        if (typeof value === 'string') return value.split(',').map(s => s.trim());
                        return [];
                    };

                    // Keys we want to merge instead of fully replacing the section
                    const MERGE_ARRAY_KEYS = [
                        'FILE_TYPES',
                        'DOMAIN_KEYWORDS',
                        'AUTO_DOMAIN_KEYWORDS',
                        'USER_ADDED_KEYWORDS'
                    ];
                    const MERGE_KEYS = [...MERGE_ARRAY_KEYS, 'SELECTED_N_TOP'];

                    // Merge each top-level section
                    for (const section in loadedConfig) {
                        if (!Object.hasOwn(loadedConfig, section)) continue;
                        const loadedValue = loadedConfig[section];
                        const currentValue = this.formConfig[section];

                        // 1. Array-to-array
                        if (Array.isArray(currentValue) && Array.isArray(loadedValue)) {
                            this.formConfig[section] = [...loadedValue];
                            continue;
                        }

                        // 2. Deep-merge objects
                        if (
                            typeof currentValue === 'object' && currentValue !== null &&
                            typeof loadedValue === 'object' && loadedValue !== null &&
                            !Array.isArray(currentValue) && !Array.isArray(loadedValue)
                        ) {
                            this.deepMerge(currentValue, loadedValue);
                            continue;
                        }

                        // 3. Fallback assignment with type-mismatch warning
                        if (typeof currentValue !== typeof loadedValue) {
                            console.warn(
                                `Section ${section} type mismatch: Frontend=${typeof currentValue}, Backend=${typeof loadedValue}`
                            );
                        }

                        // 4. Special-case merge of certain env/document keys (including SELECTED_N_TOP)
                        if (
                            (section === 'document' || section === 'env') &&
                            Object.keys(loadedValue).some(key => MERGE_KEYS.includes(key))
                        ) {
                            for (const key in loadedValue) {
                                if (MERGE_ARRAY_KEYS.includes(key)) {
                                    // always an array
                                    this.formConfig[section][key] = ensureArray(loadedValue[key]);
                                } else if (key === 'SELECTED_N_TOP') {
                                    // preserve numeric top-N setting
                                    this.formConfig[section][key] = loadedValue[key];
                                } else {
                                    // any other field in this section
                                    this.formConfig[section][key] = loadedValue[key];
                                }
                            }
                        } else {
                            // fully replace the section
                            this.formConfig[section] = loadedValue;
                        }
                    }

                    // Ensure critical nested arrays exist
                    const arrayFields = {
                        document: ['FILE_TYPES'],
                        env: ['DOMAIN_KEYWORDS', 'AUTO_DOMAIN_KEYWORDS', 'USER_ADDED_KEYWORDS']
                    };
                    for (const [sec, fields] of Object.entries(arrayFields)) {
                        if (!this.formConfig[sec]) this.formConfig[sec] = {};
                        for (const fld of fields) {
                            this.formConfig[sec][fld] = ensureArray(this.formConfig[sec][fld]);
                        }
                    }

                    // Populate keyword controls as before...
                    const autoKeys = this.formConfig.env.AUTO_DOMAIN_KEYWORDS;
                    this.autoDomainKeywordsList = [...autoKeys];
                    this.autoDomainKeywords = autoKeys.join(', ');

                    // **Updated:** apply SELECTED_N_TOP instead of SELECTED_N_TOP
                    if (this.formConfig.env.SELECTED_N_TOP != null) {
                        this.selectedTopN = this.formConfig.env.SELECTED_N_TOP;
                    }

                    this.allAutoDomainKeywordsList = Array.isArray(this.formConfig.env.AUTO_DOMAIN_KEYWORDS) ?
                        [...this.formConfig.env.AUTO_DOMAIN_KEYWORDS] : [];
                    try {
                        this.updateTopNOptions();
                        console.log(`TopN options updated with ${this.topNOptions.length} values`);
                    } catch (e) {
                        console.error("Error updating TopN options:", e);
                    }
                    
                    // this.envUserKeywordsString = this.formConfig.env.USER_ADDED_KEYWORDS.join(', ');
                    this.envUserKeywordsString = (this.formConfig.env.USER_ADDED_KEYWORDS || []).join(', ');
                    this._markConfigClean();
                    console.log("Config loaded, keywords populated, and state marked clean");

                } catch (error) {
                    console.error("Config load failed:", error);
                    this.showToast(`Config load error: ${error.message}`, 'error');
                    // reset to safe defaults...
                    this.formConfig.document.FILE_TYPES = [];
                    this.formConfig.env.DOMAIN_KEYWORDS = [];
                    this.formConfig.env.AUTO_DOMAIN_KEYWORDS = [];
                    this.formConfig.env.USER_ADDED_KEYWORDS = [];
                    this.autoDomainKeywordsList = [];
                    this.autoDomainKeywords = '';
                    throw error;
                }
            },

                    // Add deepMerge helper in your component
                // Deep‐merge two plain‐object structures
                deepMerge(target, source) {
                    for (const [key, sourceVal] of Object.entries(source)) {
                        const targetVal = target[key];

                        if (this.isPlainObject(sourceVal) && this.isPlainObject(targetVal)) {
                            this.deepMerge(targetVal, sourceVal);
                        } else if (Array.isArray(sourceVal) && Array.isArray(targetVal)) {
                            // override arrays
                            target[key] = [...sourceVal];
                        } else {
                            // clone objects/arrays to avoid shared refs
                            target[key] = this.isPlainObject(sourceVal)
                                ? { ...sourceVal }
                                : Array.isArray(sourceVal)
                                    ? [...sourceVal]
                                    : sourceVal;
                        }
                    }
                },

                // Check for a plain object
                isPlainObject(v) {
                    return Object.prototype.toString.call(v) === '[object Object]';
                },

                // Example of another helper method
                updateConfigWithKeywords(keywords) {
                    console.log("[Update Config] New keywords:", keywords);
                    this.formConfig.env.USER_ADDED_KEYWORDS = keywords;
                    this.showToast(`Configuration updated with ${keywords.length} keywords.`, 'success');
                },

            updateTopNOptions() {
                const max = this.allAutoDomainKeywordsList.length;
                this.topNOptions = Array.from(
                    { length: Math.ceil(max / 10) },
                    (_, i) => Math.min((i + 1) * 10, max)
                );
                if (max > 0 && max % 10 !== 0 && !this.topNOptions.includes(max)) {
                    this.topNOptions.push(max);
                }
                this.topNOptions.sort((a, b) => a - b);
                this.selectedTopN = this.topNOptions[0] || 0;
            },


                    async fetchPresets() {
                        console.log("Fetching presets from /list_presets...");
                        try {
                            const response = await fetch('/list_presets'); // MUST EXIST IN FLASK
                            if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                            this.presets = await response.json();
                            console.log("Presets loaded via API:", Object.keys(this.presets).length);
                        }
                        catch (error) {
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
                            const response = await fetch(`/apply_preset/${presetName}`,
                                {
                                    method: 'POST'
                                }); // MUST EXIST IN FLASK
                            if (!response.ok) {
                                let errorData = {
                                    error: `HTTP error ${response.status}`
                                };
                                try {
                                    errorData = await response.json();
                                }
                                catch (e) { }
                                throw new Error(errorData.error || errorData.message || `Failed to apply preset.`);
                            }
                            const result = await response.json(); // Expect { success: true, config: { ... } }

                            if (!result.success || !result.config) {
                                throw new Error(result.message || "Backend didn't return updated config.");
                            }

                            // Update local formConfig with the config returned by the backend
                            for (const section in result.config) {
                                if (this.formConfig[section]) {
                                    Object.assign(this.formConfig[section], result.config[section]);
                                }
                                else {
                                    this.formConfig[section] = result.config[section];
                                }
                            }

                            // Explicitly update UI state variables from the new config
                            if (result.config.retrieval && result.config.retrieval.SELECTED_N_TOP !== undefined) {
                                this.selectedNTop = result.config.retrieval.SELECTED_N_TOP;
                            }
                            
                            if (result.config.paths) {
                                this.formConfig.paths = { ...result.config.paths };
                            }

                            // Add this block to refresh auto domain keywords
                            if (this.formConfig.env && Array.isArray(this.formConfig.env.AUTO_DOMAIN_KEYWORDS)) {
                                // Update the auto domain keywords lists
                                this.allAutoDomainKeywordsList = [...this.formConfig.env.AUTO_DOMAIN_KEYWORDS];
                                this.autoDomainKeywordsList = [...this.allAutoDomainKeywordsList];
                                this.activeAutoDomainKeywordsSet = new Set(this.autoDomainKeywordsList);

                                this.updateTopNOptions();

                                // Log for debugging
                                console.log("Updated auto domain keywords after preset application:", this.allAutoDomainKeywordsList.length);
                            }

                            this._markConfigClean();

                            // Ensure arrays are correct after update
                            this.formConfig.document.FILE_TYPES = Array.isArray(this.formConfig.document
                                .FILE_TYPES) ? this.formConfig.document.FILE_TYPES : [];
                            // ... repeat for other arrays ...

                            this.showToast(`Preset '${presetName}' applied.`, 'success');
                            this.selectedPresetName = presetName; // Ensure dropdown reflects applied preset
                            this.statusMessage = 'Idle';

                        }
                        catch (error) {
                            console.error('Apply Preset Error:', error);
                            this.statusMessage = 'Preset Error';
                            this.showToast(`Error applying preset: ${error.message}`, 'error');
                            this.selectedPresetName = ''; // Clear dropdown selection on error
                            setTimeout(() => {
                                if (!this.isLoading()) this.statusMessage = 'Idle';
                            }, 3000);
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
                            console.warn('[Button Click] Save Preset ignored, already loading:', this
                                .statusMessage);
                            if (this.showToast) this.showToast("Please wait for the current operation to finish.",
                                "info");
                            return; // Stop if already busy
                        }

                        console.log(`[API Call] Saving current config as preset: ${presetName}`);
                        this.statusMessage = 'Saving Preset...'; // Set loading state

                        // Create a deep copy of the current config state to send
                        // Use try/catch for JSON operations just in case formConfig is weird
                        let configToSave;
                        try {
                            configToSave = JSON.parse(JSON.stringify(this.formConfig));
                        }
                        catch (jsonError) {
                            console.error("[savePreset] Error preparing config data:", jsonError);
                            if (this.showToast) this.showToast("Internal error preparing configuration data.",
                                "error");
                            this.statusMessage = 'Preset Error'; // Set error state
                            return; // Stop if data cannot be prepared
                        }

                        try {
                            // Make the API call to the backend
                            const response = await fetch('/save_preset',
                                { // Ensure Flask POST endpoint exists and accepts JSON
                                    method: 'POST',
                                    headers:
                                    {
                                        'Content-Type': 'application/json'
                                    },
                                    // Send preset name and the config object
                                    body: JSON.stringify(
                                        {
                                            preset_name: presetName,
                                            config: configToSave
                                        })
                                });

                            const result = await response.json(); // Always try to parse JSON response

                            // Check if the request was successful
                            if (!response.ok || !result.success) {
                                // Throw an error to be caught by the catch block
                                throw new Error(result.error || result.message ||
                                    `HTTP error! Status: ${response.status}`);
                            }

                            // --- Success Path ---
                            if (this.showToast) this.showToast(result.message || `Preset '${presetName}' saved.`,
                                'success');
                            this.newPresetName = ''; // Clear the input field
                            await this.fetchPresets(); // Refresh the preset list dropdown
                            this.selectedPresetName = presetName; // Select the newly saved preset in the dropdown

                            // statusMessage will be reset in the finally block on success

                        }
                        catch (error) {
                            // --- Error Path ---
                            console.error('[API Error] Save Preset FAILED:', error);
                            this.statusMessage = 'Preset Error'; // Set specific error status
                            if (this.showToast) this.showToast(`Error saving preset: ${error.message}`, 'error');
                            // Optional: Delay before resetting error status to Idle
                            setTimeout(() => {
                                if (this.statusMessage === 'Preset Error') this.statusMessage = 'Idle';
                            }, 3000);

                        }
                        finally {
                            // --- CORRECTED finally block ---
                            // Reset status *immediately* ONLY IF it's still 'Saving Preset...'
                            // This handles the success case correctly and avoids interfering with the error case.
                            if (this.statusMessage === 'Saving Preset...') {
                                this.statusMessage = 'Idle';
                                console.log("[savePreset finally] Status reset to Idle after success.");
                            }
                            else {
                                // Log if status was already changed (e.g., by the catch block)
                                console.log("[savePreset finally] Status not 'Saving Preset...', current:", this
                                    .statusMessage);
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
                            console.warn('[Button Click] Delete Preset ignored, already loading:', this
                                .statusMessage);
                            if (this.showToast) this.showToast("Please wait for the current operation to finish.",
                                "info");
                            return;
                        }

                        // Use the confirmation modal
                        this.requireConfirmation(
                            {
                                title: 'Delete Preset',
                                message: `Are you sure you want to permanently delete the preset "${presetName}"?`,
                                confirmButtonClass: 'bg-red-600 hover:bg-red-700', // Red confirm button
                                onConfirm: async () => { // The actual deletion logic
                                    console.log(`[Confirm Action] Deleting preset: ${presetName}`);
                                    this.statusMessage = 'Deleting Preset...';
                                    try {
                                        // Make the API call to the backend
                                        // Use encodeURIComponent in case preset names have special characters
                                        const response = await fetch(
                                            `/delete_preset/${encodeURIComponent(presetName)}`,
                                            {
                                                method: 'DELETE' // Use DELETE HTTP method
                                            });

                                        const result = await response
                                            .json(); // Expect { success: true/false, message: '...' }

                                        // Check if the request was successful
                                        if (!response.ok || !result.success) {
                                            throw new Error(result.error || result.message ||
                                                `HTTP error! Status: ${response.status}`);
                                        }

                                        // --- Success Path ---
                                        if (this.showToast) this.showToast(result.message ||
                                            `Preset '${presetName}' deleted.`, 'success');
                                        await this.fetchPresets(); // Refresh the preset list dropdown
                                        this.selectedPresetName = ''; // Clear the selection as the preset is gone

                                        // statusMessage will be reset in finally block

                                    }
                                    catch (error) {
                                        // --- Error Path ---
                                        console.error('[API Error] Delete Preset FAILED:', error);
                                        this.statusMessage = 'Delete Error';
                                        if (this.showToast) this.showToast(
                                            `Error deleting preset: ${error.message}`, 'error');
                                        // Optional: Delay before resetting error status
                                        setTimeout(() => {
                                            if (this.statusMessage === 'Delete Error') this.statusMessage =
                                                'Idle';
                                        }, 3000);

                                    }
                                    finally {
                                        // --- finally block ---
                                        // Reset status ONLY IF it's still 'Deleting Preset...'
                                        if (this.statusMessage === 'Deleting Preset...') {
                                            this.statusMessage = 'Idle';
                                            console.log("[deletePreset finally] Status reset to Idle after attempt.");
                                        }
                                        else {
                                            console.log(
                                                "[deletePreset finally] Status not 'Deleting Preset...', current:",
                                                this.statusMessage);
                                        }
                                        // --- END finally block ---
                                    }
                                }, // End onConfirm
                                onCancel: () => {
                                    console.log(`[Confirm Action] Cancelled deletion for preset: ${presetName}`);
                                }
                            }); // End requireConfirmation
                    }, // End of deletePresetWithConfirmation

                    
                    async fetchCentroidStats() {
                        try {
                            const resp = await fetch('/api/centroid');
                            if (resp.ok) {
                                this.centroidStats = await resp.json();
                            } else {
                                console.error("Failed to fetch centroid stats:", await resp.text());
                            }
                        } catch (error) {
                            console.error("Error fetching centroid stats:", error);
                        }
                    },


                    async fetchQueryCentroidInsight(queryEmbedding) {
                        const resp = await fetch('/api/centroid/query_insight', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ vector: queryEmbedding })
                        });
                        if (resp.ok) this.queryCentroidInsight = await resp.json();
                    },

            async fetchAutoDomainKeywords() {
                try {
                    console.log("[Init API] Starting fetchAutoDomainKeywords...");
                    const res = await fetch('/get_auto_domain_keywords');
                    const data = await res.json();

                    // Check if data.keywords exists and is an array
                    this.allAutoDomainKeywordsList = Array.isArray(data.keywords) ? data.keywords : [];

                    // 2. Comma-string for the textarea display
                    this.autoDomainKeywords = this.allAutoDomainKeywordsList.length
                        ? this.allAutoDomainKeywordsList.join(', ')
                        : '[No auto domain keywords found]';

                    // 3. Set for toggling individual keywords
                    this.activeAutoDomainKeywordsSet = new Set(this.allAutoDomainKeywordsList);

                    // 4. Build options using the helper method
                    this.updateTopNOptions();
                    this.applyTopNKeywords();

                    console.log("[Init API] Finished fetchAutoDomainKeywords with",
                        this.allAutoDomainKeywordsList.length, "keywords");
                }
                catch (error) {
                    console.error('Failed to fetch auto domain keywords:', error);
                    this.allAutoDomainKeywordsList = [];
                    this.autoDomainKeywords = '[Error loading auto domain keywords]';
                    this.activeAutoDomainKeywordsSet = new Set();

                    // Still call updateTopNOptions to ensure UI is consistent
                    this.updateTopNOptions();
                }
            },


                    // Send the active keywords to backend (update config/file)
            async syncActiveKeywordsToBackend() {
                try {
                    console.log("Syncing keywords to backend:", Array.from(this.activeAutoDomainKeywordsSet));
                    const response = await fetch('/update_auto_domain_keywords', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            keywords: Array.from(this.activeAutoDomainKeywordsSet),
                            target_field: "AUTO_DOMAIN_KEYWORDS"  // Add this line to specify the target field
                        })
                    });
                    const result = await response.json();
                    if (!result.success) {
                        console.error("Backend sync failed:", result.error);
                        throw new Error(result.error || "Unknown error");
                    }
                    console.log("Keywords synced successfully");
                } catch (e) {
                    console.error("Sync error:", e);
                    this.showToast(`Failed to update keywords: ${e.message}`, "error");
                }
            },
            clearChatWithConfirmation() {
                if (confirm('Are you sure you want to clear the chat history?')) {
                    this.chatHistory = []; // Change from this.messages to this.chatHistory

                    // If saving to localStorage
                    localStorage.removeItem('chatMessages');

                    // If using server-side storage, add an API call
                    fetch('/clear-chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    
                });
                }

}}) // <-- End of Alpine.data object
)}); // <-- End of event listener

function addTooltips() {
    // Create a function to add tooltips to labels
    function createTooltip(label, key) {
        // Only proceed if we have content for this key
        if (!tooltipContent[key]) return;

        // Create wrapper
        const wrapper = document.createElement('span');
        wrapper.className = 'tooltip-wrapper';

        // Create icon
        const icon = document.createElement('span');
        icon.className = 'tooltip-icon';
        icon.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <path d="M12 16v-4"></path>
        <path d="M12 8h.01"></path>
      </svg>`;

        // Create tooltip content
        const tooltip = document.createElement('span');
        tooltip.className = 'tooltip-content';
        tooltip.textContent = tooltipContent[key];

        // Assemble tooltip
        wrapper.appendChild(icon);
        wrapper.appendChild(tooltip);

        // Add tooltip after the label text
        label.appendChild(wrapper);
    }

    // Process all labels with data-tooltip attribute
    document.querySelectorAll('[data-tooltip]').forEach(label => {
        const key = label.getAttribute('data-tooltip');
        createTooltip(label, key);
    });

}



// Consolidated listener for non-Alpine DOM setup
let domContentLoadedFired_setup = false; // Use a specific flag name

document.addEventListener('DOMContentLoaded', function () {
    if (domContentLoadedFired_setup) {
        console.warn('[DOM Setup] Consolidated listener fired AGAIN. Preventing re-execution.');
        return;
    }
    domContentLoadedFired_setup = true;
});

// Final log to confirm script parsing
console.log('static/ragapp.js parsed successfully.');

// --- END static/ragapp.js ---//
