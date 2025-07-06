// --- START static/ragapp.js ---
// Define Alpine component at the right time, no global ragApp, no DOMContentLoaded flag

// ragapp.js
document.addEventListener('alpine:init', () => {
    console.log('[Alpine Event] alpine:init fired. Defining ragApp component...');
    Alpine.data('ragApp', () => ({
        
        collections: [],
        selected: null,
        status: 'Loading collections…',
        envUserKeywordsString: '',
        // ─── CORE STATE ───
        docFreqMode: 'absolute',   
        totalDocs: 0,              
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
        clearChat() {this.chatHistory = [];        },
        autoDomainKeywords: '',
        lastAutoDomainKeywordsList: [],
        centroidStats: { centroid: null, meta: null},
        histogramUrl: '/static/placeholder.png',
        queryCentroidInsight: { similarity: 0, distance: 0 },
        centroidUpdateMode: 'auto',
        centroidAutoThreshold: 5,
        inspectModal: {
            show: false,
            data: null
          },

        // ─── CONFIGURATION ───
        formConfig: {
            
            ingestion: {
                CENTROID_AUTO_THRESHOLD: 0,
                CENTROID_DIVERSITY_THRESHOLD: 0,
                CENTROID_UPDATE_MODE: "auto",
                MIN_QUALITY_SCORE: 0.3
                },

            security: {
                SANITIZE_INPUT: true,
                RATE_LIMIT: 100,
                API_TIMEOUT: 30,
                CACHE_ENABLED: true
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
                PERFORM_TECHNICAL_VALIDATION: true,
                WEAVIATE_HOST: '',
                WEAVIATE_HTTP_PORT: 0,
                WEAVIATE_GRPC_PORT: 0
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
                EXTERNAL_API_MODEL_NAME: null
            },
            document: {
                CHUNK_SIZE: 1000,
                CHUNK_OVERLAP: 100,
                FILE_TYPES: [],
                PARSE_TABLES: true,
                GENERATE_SUMMARY: false
            },
            paths: {
                DOCUMENT_DIR: './data',
                //DOMAIN_CENTROID_PATH: './domain_centroid.npy'
                CENTROID_DIR: './centroids'
            },
            env: {
                DOMAIN_KEYWORDS: [],
                AUTO_DOMAIN_KEYWORDS: [],
                USER_ADDED_KEYWORDS: [],
                SELECTED_N_TOP: null
            },
            pipeline: {
                max_history_turns: 5
            },
            domain_keyword_extraction: {
            // persisted mode & two inputs
            min_doc_freq_mode: 'absolute',
            min_doc_freq_abs: null,
            min_doc_freq_frac: null,
            min_doc_freq: 2,
            extraction_diversity: 0.7,
            no_pos_filter: false
        }
          },
        
          emptyCentroid() {
            return {
                centroid: null,
                shape: null,
                path: null,
                meta: null,
                loaded: false
            };
        },



        // ─── LIFECYCLE METHODS ───
        init() {
            console.log("[UI Init] Starting UI initialization...");

            if (this.initCalled) {
                console.warn("[UI Init] Skipping duplicate init() call.");
                return;
            }
            this.initCalled = true;

            this.statusMessage = 'Initializing...';
            this.status = 'Loading collections...';

            this.checkApiKeys();

            this.isLoadingConfig = true;
            this.isLoadingPresets = true;
            this.isLoadingAutoKeywords = true;

            

            Promise.all([
                fetch('/api/config').then(r => r.json()),
                fetch('/api/collections').then(r => r.json())
            ])
                .then(([configData, collectionsData]) => {
                    // Remove presets before assigning
                    const { presets, ...configOnly } = configData || {};
                    this.formConfig = configOnly;
                    this.presets = presets || {};

                    const arr = Array.isArray(collectionsData)
                        ? collectionsData
                        : (collectionsData?.collections || []);

                    if (!Array.isArray(arr)) {
                        console.error('Invalid /api/collections payload:', collectionsData);
                        throw new TypeError('Expected array or { collections: array }');
                    }

                    this.collections = arr;

                    // Load from localStorage
                    const storedCollection = localStorage.getItem('selected_collection');
                    const collectionFromConfig = this.formConfig?.retrieval?.COLLECTION_NAME || null;
                    const aliasFromConfig = this.formConfig?.retrieval?.WEAVIATE_ALIAS || null;
                    const isStoredValid = storedCollection && arr.includes(storedCollection);
                    const isConfigValid = collectionFromConfig && arr.includes(collectionFromConfig);

                    if (isStoredValid) {
                        console.log(`[Init] Restoring collection from localStorage: ${storedCollection}`);
                        this.selected = storedCollection;
                    } else if (isConfigValid) {
                        console.log(`[Init] Using config-specified collection: ${collectionFromConfig}`);
                        this.selected = collectionFromConfig;
                    } else {
                        console.warn(`[Init] No valid stored/config collection — defaulting to: ${arr[0] || 'none'}`);
                        this.selected = arr[0] || null;
                    }

                    // Store alias
                    const aliasOverride = localStorage.getItem('selected_instance_alias');
                    if (aliasOverride) {
                        this.selectedInstanceAlias = aliasOverride;
                    } else if (aliasFromConfig) {
                        this.selectedInstanceAlias = aliasFromConfig;
                        localStorage.setItem('selected_instance_alias', aliasFromConfig);
                    } else {
                        this.selectedInstanceAlias = 'main';
                    }

                    if (this.selected) {
                        this.status = `Loading centroid for '${this.selected}'…`;
                        this.loadCentroid();
                    } else {
                        this.status = 'No collections available.';
                    }

                    if (!this.presetsLoaded) {
                        this.presetsLoaded = true;
                        this.fetchPresets?.();
                    }

                    if (!this.weaviateInstancesLoaded) {
                        this.weaviateInstancesLoaded = true;
                        this.fetchWeaviateInstances?.();
                    }

                    if (!this.savedChatsLoaded) {
                        this.savedChatsLoaded = true;
                        this.fetchSavedChats?.();
                    }

                    if (!this.autoKeywordsLoaded) {
                        this.autoKeywordsLoaded = true;
                        this.fetchAutoDomainKeywords?.();
                    }
                })
                .catch(err => {
                    console.error('Initialization error in Promise.all:', err);
                    this.status = 'Initialization error.';
                })
                .finally(() => {
                    this.isLoadingConfig = false;
                    this.isLoadingPresets = false;
                    this.isLoadingAutoKeywords = false;
                });

            if (!this.formConfig.domain_keyword_extraction) {
                this.formConfig.domain_keyword_extraction = {
                    diversity: 0,
                    extraction_diversity: 0,
                    top_n_per_doc: 1,
                    min_doc_freq_abs: 0,
                    min_doc_freq_frac: 0
                };
            }
            

            const kde = this.formConfig.domain_keyword_extraction;
            if (kde.extraction_diversity == null) {
                kde.extraction_diversity = typeof kde.diversity === 'number' ? kde.diversity : 0;
            }

            const clamps = [
                ['formConfig.ingestion.CENTROID_AUTO_THRESHOLD', 0, 100],
                ['formConfig.domain_keyword_extraction.top_n_per_doc', 1, 50],
                ['formConfig.document.CHUNK_SIZE', 1, null],
                ['formConfig.document.CHUNK_OVERLAP', 0, null],
                ['formConfig.model.MAX_TOKENS', 1, null],
                ['formConfig.domain_keyword_extraction.min_doc_freq_abs', 0, null],
                ['formConfig.domain_keyword_extraction.min_doc_freq_frac', 0, 1],
                ['formConfig.security.RATE_LIMIT', 0, null],
                ['formConfig.security.API_TIMEOUT', 0, null],
                ['formConfig.retrieval.K_VALUE', 1, null],
            ];

            for (const [path, min, max] of clamps) {
                this.$watch(path, v => {
                    if (typeof v === 'number') {
                        let clamped = v;
                        if (min !== null && v < min) clamped = min;
                        if (max !== null && v > max) clamped = max;
                        if (clamped !== v) {
                            this.showToast(
                                `${path.split('.').pop()} must be between ${min ?? '-∞'} and ${max ?? '∞'}.`,
                                'error'
                            );
                            const keys = path.split('.');
                            let obj = this;
                            while (keys.length > 1) obj = obj[keys.shift()];
                            obj[keys[0]] = clamped;
                        }
                    }
                });
            }
        },
        


        formatKey(key) {
            if (!key) return '';
            return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        },

        formatValue(value) {
            if (typeof value === 'number') {
                return value.toFixed(4);
            }
            return String(value);
        },


        
        async loadCentroid() {
            const collection = this.selected;
            const instance = localStorage.getItem('selected_instance_alias') || 'Main';

            if (!collection || !instance) return;

            console.warn(`[loadCentroid] for '${collection}' from instance '${instance}'`);
            this.status = `Loading centroid for '${instance}_${collection}'…`;

            try {
                const res = await fetch(`/api/centroid?collection=${encodeURIComponent(collection)}&instance=${encodeURIComponent(instance)}`);
                const text = await res.text();
                const contentType = res.headers.get("content-type") || "";

                if (!res.ok || !contentType.includes("application/json")) {
                    console.error('[Load Error]', text);
                    this.status = `Load failed: unexpected response`;
                    this.centroidStats.meta = null;
                    this.histogramUrl = '/static/placeholder.png';
                    return;
                }

                let json;
                try {
                    json = JSON.parse(text);
                } catch (err) {
                    console.error('[Load JSON error]', text);
                    this.status = 'Load failed: invalid JSON';
                    this.centroidStats.meta = null;
                    this.histogramUrl = '/static/placeholder.png';
                    return;
                }

                if (!json || !Array.isArray(json.shape) || !json.shape.length) {
                    this.status = 'Centroid not found. Run calculate?';
                    this.centroidStats.meta = null;
                    this.histogramUrl = '/static/placeholder.png';
                    this.centroidStats.centroid = false;
                    return;
                }

                this.centroidStats = {
                    centroid: true,
                    shape: json.shape,
                    path: json.path,
                    meta: null,
                    loaded: true
                };

                this.histogramUrl = `/centroid_histogram.png?collection=${encodeURIComponent(collection)}&instance=${encodeURIComponent(instance)}&t=${Date.now()}`;
                console.warn(`[loadCentroid] Centroid loaded for '${collection}', fetching stats`);
                await this.fetchCentroidStats();

            } catch (err) {
                console.error('[Load Fetch error]', err);
                this.status = 'Error loading centroid.';
                this.centroidStats.meta = null;
                this.histogramUrl = '/static/placeholder.png';
                this.centroidStats.centroid = false;
            }
        },
        
        async recalculate() {
            if (!this.selected) return;

            const instance = this.selectedInstance || 'Main'; // fallback if undefined

            console.log('[Recalc] skipCollection:', true, 'payload.retrieval.COLLECTION_NAME:', this.selected);
            await this.saveConfig({ skipCollection: true });

            this.status = `Recalculating centroid for '${this.selected}'…`;

            try {
                const res = await fetch(
                    `/api/centroid?collection=${encodeURIComponent(this.selected)}&instance=${encodeURIComponent(instance)}`,
                    { method: 'POST' }
                );

                const text = await res.text();

                if (!res.ok) {
                    console.error('[Recalc Error]', text);
                    this.status = `Recalc failed: ${res.status}`;
                    this.centroidStats = this.emptyCentroid();
                    this.histogramUrl = '/static/placeholder.png';
                    return;
                }

                let json;
                try {
                    json = JSON.parse(text);
                } catch {
                    console.error('[Recalc JSON error]', text);
                    this.status = 'Recalc failed: invalid JSON';
                    this.centroidStats = this.emptyCentroid();
                    this.histogramUrl = '/static/placeholder.png';
                    return;
                }

                if (json.ok) {
                    this.status = `Recalculated! New shape ${json.shape} at ${json.path}`;
                    this.centroidStats = this.emptyCentroid();
                    this.histogramUrl = '/static/placeholder.png';

                    console.log("[Recalc] Loading centroid...");
                    await this.loadCentroid();

                    console.log("[Recalc] Scheduling centroidStats fetch");
                    setTimeout(() => {
                        const root = document.querySelector('[x-data]');
                        const ctx = Alpine.$data(root);
                        if (typeof ctx.fetchCentroidStats === 'function') {
                            console.warn('[CTX FIX] Forcing Alpine-bound fetchCentroidStats');
                            ctx.fetchCentroidStats();
                        } else {
                            console.error('[CTX FIX] Alpine fetchCentroidStats is missing');
                        }
                    }, 200);
                } else {
                    this.status = `Recalc error: ${json.error || 'Centroid could not be calculated or saved.'}`;
                    this.centroidStats = this.emptyCentroid();
                    this.histogramUrl = '/static/placeholder.png';
                }
            } catch (err) {
                console.error('[Recalc Fetch error]', err);
                this.status = 'Error during recalculation.';
                this.centroidStats = this.emptyCentroid();
                this.histogramUrl = '/static/placeholder.png';
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

        
        

        applyTopNKeywords() {
            if (this.isSavingTopN) return;

            const n = this.selectedTopN;

            // Skip if already applied
            if (n === this.lastSavedTopN) {
                console.log(`[TopN] Skipped — TopN=${n} already saved.`);
                return;
            }

            console.log(`[TopN] Applying top ${n} keywords`);
            this.lastAutoDomainKeywordsList = [...this.autoDomainKeywordsList];
            this.autoDomainKeywordsList = this.allAutoDomainKeywordsList.slice(0, n);
            this._updateActiveSet();
            this.syncActiveKeywordsToBackend();

            this.saveTopNToConfig(n);
        },
        
        async saveTopNToConfig(topN) {
            if (this.isSavingTopN) return;
            this.isSavingTopN = true;

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
                    this.lastSavedTopN = topN;
                    this.showToast("Top-N saved", "success");
                }
            } catch (e) {
                console.error("[Config] Error saving TopN to config:", e);
                this.showToast(`Failed to save Top-N: ${e}`, "error");
            } finally {
                this.isSavingTopN = false;
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

        promptForChatName(defaultName) {
            return new Promise((resolve) => {
                this.confirmationModal = {
                    show: true,
                    title: "Save Chat",
                    message: "", // We'll put the input in the modal
                    confirmButtonClass: "bg-green-600 hover:bg-green-700",
                    onConfirm: () => {
                        const nameInput = document.getElementById('chatNameInput');
                        const name = nameInput?.value?.trim();
                        this.confirmationModal.show = false;
                        resolve(name || null);
                    },
                    onCancel: () => {
                        this.confirmationModal.show = false;
                        resolve(null);
                    }
                };

                this.$nextTick(() => {
                    const input = document.getElementById('chatNameInput');
                    if (input) input.focus();
                });
            });
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

        async fetchCollections() {
            try {
                const response = await fetch('/api/collections');
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const data = await response.json();
                const arr = Array.isArray(data) ? data : (data?.collections || []);
                if (!Array.isArray(arr)) throw new Error("Invalid collections payload");
                this.collections = arr;
            } catch (err) {
                console.error("[fetchCollections] Error:", err);
                this.collections = [];
                this.showToast?.("Failed to load collections", "error");
            }
              },

        async createNewCollection() {
            const name = prompt("Enter new collection name:");
            if (!name || !name.trim()) return;

            this.statusMessage = `Creating collection '${name}'...`;
            try {
                const res = await fetch("/create_collection", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ name: name.trim() })
                });
                const result = await res.json();
                if (!res.ok || !result.success) {
                    const msg = result.error || `Error ${res.status}`;
                    this.showToast(`Creation failed: ${msg}`, "error");
                } else {
                    this.showToast(`Collection '${name}' created.`, "success");
                    // Refresh collections
                    await this.fetchCollections();
                    // Auto-select the new collection
                    this.selected = name;
                }
            } catch (e) {
                console.error("Error creating collection:", e);
                this.showToast(`Error: ${e.message}`, "error");
            } finally {
                this.statusMessage = "Idle";
            }
        },
            
        async syncActiveKeywordsToBackend() {
            const keywords = Array.from(this.activeAutoDomainKeywordsSet || []);

            // Skip if same keywords as last sync
            if (JSON.stringify(keywords) === JSON.stringify(this.lastSyncedAutoKeywords || [])) {
                console.log("[Keywords] No change. Skipping backend sync.");
                return;
            }

            if (this.isSyncingKeywords) {
                console.warn("[Keywords] Sync already in progress. Skipping.");
                return;
            }

            this.isSyncingKeywords = true;

            try {
                console.log("[Keywords] Syncing to backend:", keywords);
                const response = await fetch('/update_auto_domain_keywords', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        keywords,
                        target_field: "AUTO_DOMAIN_KEYWORDS"
                    })
                });

                const result = await response.json();
                if (!result.success) {
                    console.error("[Keywords] Backend sync failed:", result.error);
                    throw new Error(result.error || "Unknown error");
                }

                console.log("[Keywords] Synced successfully");
                this.lastSyncedAutoKeywords = keywords;
            } catch (e) {
                console.error("[Keywords] Sync error:", e);
                this.showToast(`Failed to update keywords: ${e.message}`, "error");
            } finally {
                this.isSyncingKeywords = false;
            }
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
                    lastSavedTopN: null,
                    lastSyncedAutoKeywords: [],
                    isSyncingKeywords: false,
                    // …other state…
                }
            },
              
        async saveConfig({ skipCollection = false } = {}) {
            if (this.isLoading() && !skipCollection) {
                console.warn('[saveConfig] Ignored, already loading:', this.statusMessage);
                if (this.showToast) this.showToast("Please wait for the current operation to finish.", "info");
                return false;
            }

            console.log(
                "[saveConfig] Preparing payload...",
                "\nformConfig keys:", Object.keys(this.formConfig),
                "\nretrieval keys:", Object.keys(this.formConfig.retrieval),
                "\nFull config:\n", JSON.stringify(this.formConfig, null, 2)
            );

            this.statusMessage = 'Saving config';

            const dke = this.formConfig.domain_keyword_extraction;
            dke.min_doc_freq_mode = this.docFreqMode;

            if (this.docFreqMode === 'absolute') {
                dke.min_doc_freq = dke.min_doc_freq_abs ?? dke.min_doc_freq;
            } else {
                dke.min_doc_freq = (this.totalDocs && dke.min_doc_freq_frac != null)
                    ? Math.ceil(dke.min_doc_freq_frac * this.totalDocs)
                    : dke.min_doc_freq;
            }

            try {
                const payload = JSON.parse(JSON.stringify(this.formConfig));
                // --- Bulletproof numeric coercion ---
                if (payload.pipeline && payload.pipeline.max_history_turns != null) {
                    payload.pipeline.max_history_turns = Number(payload.pipeline.max_history_turns);
                    if (isNaN(payload.pipeline.max_history_turns)) {
                        console.warn("[saveConfig] Invalid max_history_turns, falling back to 5.");
                        payload.pipeline.max_history_turns = 5;
                    }
                }

                console.log("[saveConfig] skipCollection:", skipCollection, "payload.retrieval.COLLECTION_NAME:", payload.retrieval?.COLLECTION_NAME);

                if (skipCollection && payload?.retrieval?.COLLECTION_NAME) {
                    console.warn('[saveConfig] Skipping COLLECTION_NAME during save.');
                    delete payload.retrieval.COLLECTION_NAME;
                }

                delete payload.presets;

                const response = await fetch('/save_config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    const text = await response.text();
                    throw new Error(`HTTP ${response.status}: ${text}`);
                }

                const result = await response.json();
                const msg = result.message || '';
                const isSoft = msg.includes("pipeline reload");

                if (result.success === false && !isSoft) {
                    const errorMsg = result.error || msg || "Unknown error saving config.";
                    throw new Error(errorMsg);
                }

                if (result.config) {
                    console.log("[saveConfig] Merging config from backend...");
                    console.log("[saveConfig] Backend returned config:", result.config);
                    this.deepMerge(this.formConfig, result.config);
                }

                this.showToast(msg || "Configuration saved successfully!", isSoft ? 'info' : 'success');
                this._markConfigClean();
                return true;

            } catch (error) {
                console.error('Configuration Save Error:', error);
                const msg = error?.message || String(error) || "Unknown error";
                const isSoft = msg.includes("pipeline reload");
                this.statusMessage = isSoft ? 'Idle' : 'Save Error';
                this.showToast(msg, isSoft ? 'info' : 'error');
                return !isSoft;

            } finally {
                if (this.statusMessage === 'Saving config') {
                    this.statusMessage = 'Idle';
                }
            }
        },
            
        deleteCollection(name) {
            this.requireConfirmation({
                title: "Delete Collection",
                message: `Are you sure you want to delete '${name}'? This cannot be undone.`,
                confirmButtonClass: "bg-red-600 hover:bg-red-700",
                onConfirm: () => {
                    fetch("/delete_collection", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ name })
                    })
                        .then(res => res.json())
                        .then(data => {
                            if (data.success) {
                                this.showToast?.(`Collection '${name}' deleted.`, "success");
                                this.fetchCollections();
                            } else {
                                this.showToast?.(`Error: ${data.error || "Unknown error"}`, "error");
                            }
                        });
                }
            });
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
                        console.log("[Init API] Starting fetchCollections...");
                        try {
                            await this.fetchCollections();
                            console.log("[Init API] Finished fetchCollections.");
                        } catch (e) {
                            console.error("[Init API] fetchCollections FAILED:", e);
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
                            this.totalDocs = 0;
                        }

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
            console.log('[Button Click] Extract Domain Keywords clicked. Checking for unsaved changes...');

            const callback = () => {
                this.$nextTick(() => {
                    console.log('[ConfirmUnsaved] Proceeding (deferred) to run _performRunKeywordBuilder...');
                    this._performRunKeywordBuilder();
                });
            };

            this._confirmUnsavedChanges('Extract Domain Keywords', callback);
        },
            

                    // ADD THIS METHOD: Contains the original keyword builder logic
        async _performRunKeywordBuilder() {
            if (this.isLoading()) {
                console.warn('[Keyword Builder] Ignored: already loading:', this.statusMessage);
                return;
            }

            console.log("[Keyword Builder] Running core logic...");
            this.statusMessage = 'Extracting keywords...';

            // === DOM LOOKUP - SAFE ===
            const formElement = document.getElementById('keywordBuilderForm');
            const keywordResultsDiv = formElement?.querySelector('[x-ref="keywordResults"]');
            const keywordListDiv = formElement?.querySelector('[x-ref="keywordList"]');

            console.debug("[DOM Check]", { formElement, keywordResultsDiv, keywordListDiv });

            if (!formElement || !keywordResultsDiv || !keywordListDiv) {
                console.error("[Keyword Builder] Missing required UI elements.");
                this.statusMessage = 'UI Error';
                this.showToast?.("Keyword builder UI elements missing.", "error");
                setTimeout(() => {
                    if (this.statusMessage === 'UI Error') this.statusMessage = 'Idle';
                }, 3000);
                return;
            }

            keywordResultsDiv.style.display = 'none';
            keywordListDiv.innerHTML = '';

            // === FORM DATA ===
            const formData = new FormData(formElement);
            const jsonData = {};

            try {
                for (const [key, value] of formData.entries()) {
                    if (key === 'no_pos_filter') {
                        jsonData[key] = true;
                    } else if (key === 'extraction_diversity') {
                        jsonData[key] = parseFloat(value);
                    } else if (key === 'min_doc_freq_abs') {
                        jsonData.min_doc_freq_abs = parseInt(value, 10) || null;
                    } else if (key === 'min_doc_freq_frac') {
                        const frac = parseFloat(value);
                        jsonData[key] = isNaN(frac) ? null : frac;
                        if (isNaN(frac)) console.warn(`[Keyword Builder] Invalid number for ${key}: ${value}`);
                    } else {
                        jsonData[key] = value;
                    }
                }
            } catch (formError) {
                console.error('[Keyword Builder] Error processing form data:', formError);
                this.statusMessage = 'Form Error';
                this.showToast?.("Error reading keyword settings.", "error");
                setTimeout(() => {
                    if (this.statusMessage === 'Form Error') this.statusMessage = 'Idle';
                }, 3000);
                return;
            }

            console.log("[API Call] POST /run_keyword_builder", jsonData);

            try {
                const response = await fetch('/run_keyword_builder', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(jsonData)
                });

                const data = await response.json();

                if (!response.ok || !data.success) {
                    const detail = data.error || `HTTP ${response.status}`;
                    if (data.full_error) console.error("Full Keyword Builder Error:", data.full_error);
                    throw new Error(detail + (data.details ? ` | ${data.details}` : ''));
                }

                console.log("[API Success] Extracted keywords:", data);
                keywordResultsDiv.style.display = 'block';

                if (Array.isArray(data.keywords) && data.keywords.length > 0) {
                    const html = data.keywords.map(kw =>
                        `<li class="text-xs">${kw.term}: ${kw.score.toFixed(4)}</li>`
                    ).join('');
                    keywordListDiv.innerHTML = `<ul class="list-disc list-inside">${html}</ul>`;

                    const updateConfigBtn = document.createElement('button');
                    updateConfigBtn.className = 'btn btn-success mt-3 text-xs py-1 px-2';
                    updateConfigBtn.textContent = 'Update Config with These Keywords';
                    updateConfigBtn.onclick = () => this.updateConfigWithKeywords(
                        data.keywords.map(kw => kw.term)
                    );
                    keywordListDiv.appendChild(updateConfigBtn);

                    this.showToast?.(`Extracted ${data.keywords.length} keywords.`, "success");
                    this.statusMessage = 'Keywords Extracted';
                } else {
                    keywordListDiv.innerHTML = `<p class="text-xs text-slate-600">${data.message || 'No keywords extracted.'}</p>`;
                    this.showToast?.(data.message || 'No keywords extracted.', 'info');
                    this.statusMessage = 'No Keywords Found';
                }

            } catch (error) {
                console.error('[API Error] Keyword Builder failed:', error);
                keywordListDiv.innerHTML = `<p class="text-xs text-red-600">Error: ${error.message}</p>`;
                keywordResultsDiv.style.display = 'block';
                this.showToast?.(`Keyword extraction failed: ${error.message}`, 'error', 10000);
                this.statusMessage = 'Keyword Error';
            } finally {
                setTimeout(() => {
                    const completeStates = ['Extracting keywords...', 'Keywords Extracted', 'No Keywords Found', 'Keyword Error', 'Form Error', 'UI Error'];
                    if (completeStates.includes(this.statusMessage)) {
                        this.statusMessage = 'Idle';
                    }
                }, 4000);
            }
        },
                    
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
                    confirmButtonClass: 'bg-green-600 hover:bg-green-700',
                    onConfirm: async () => {
                        console.log(`[Confirm Action] User chose Save & Proceed for ${actionName}.`);
                        const saveSuccess = await this.saveConfig();
                        window.location.reload();
                        if (saveSuccess) {
                            console.log(`[Confirm Action] Config saved. Proceeding with ${actionName}...`);
                            actionFunction();
                        } else {
                            console.warn(`[Confirm Action] Config save failed. Aborting ${actionName}.`);
                            this.statusMessage = 'Save Failed';
                            setTimeout(() => {
                                if (this.statusMessage === 'Save Failed') this.statusMessage = 'Idle';
                            }, 2000);
                        }
                    },
                    onCancel: () => {
                        console.log(`[Confirm Action] User cancelled ${actionName}.`);
                        if (this.showToast) this.showToast(`Action '${actionName}' cancelled.`, 'info');
                        // Reset status if needed
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

            const adminKey = this.adminApiKey || "";

            try {
                const response = await fetch('/api/key_status', {
                    method: 'GET',
                    headers: {
                        "X-ADMIN-KEY": adminKey,
                        "Accept": "application/json"
                    }
                });

                if (!response.ok)
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);

                const status = await response.json();
                this.apiKeyStatus.deepseek = status.deepseek || false;
                this.apiKeyStatus.openai = status.openai || false;
                this.apiKeyStatus.anthropic = status.anthropic || false;
                this.apiKeyStatus.cohere = status.cohere || false;

                console.log("[API Resp] API Key status loaded:", this.apiKeyStatus);
                this.statusMessage = "Idle";  // ✅ this line fixes it
            }
            catch (error) {
                console.error("[API Error] checkApiKeys FAILED:", error);
                this.apiKeyStatus = {
                    deepseek: false,
                    openai: false,
                    anthropic: false,
                    cohere: false
                };
                if (this.showToast)
                    this.showToast('Could not check API key status.', 'error');
                this.statusMessage = "Error";
            }
                    },

                    // --- Weaviate Instance Management ---
        async fetchWeaviateInstances() {
            console.log("[API Call] Fetching Weaviate instances from /list_weaviate_instances...");
            const LOADING_STATUS = 'Loading Weaviate instances...';
            this.statusMessage = LOADING_STATUS;

            try {
                const response = await fetch('/list_weaviate_instances');

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();

                if (!Array.isArray(data)) {
                    throw new Error("Expected array but got: " + typeof data);
                }

                this.weaviateInstances = data;
                console.log("[API Logic] Processed Weaviate instances:", this.weaviateInstances);

                // Clear status *only* if no other fetch overwrote it
                if (this.statusMessage === LOADING_STATUS) {
                    this.statusMessage = 'Idle';
                }

            } catch (error) {
                console.error("[API Error] fetchWeaviateInstances FAILED:", error);
                this.weaviateInstances = [];

                if (this.showToast) {
                    this.showToast(`Weaviate instance list failed: ${error.message}`, "error");
                }

                if (this.statusMessage === LOADING_STATUS) {
                    this.statusMessage = 'Instance fetch error';
                }
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

                if (this.showToast) {
                    this.showToast(
                        result.message || `Instance '${instanceName}' activated.`,
                        'success'
                    );
                }

                await this.loadInitialConfig();
                this.selectedInstanceAlias = instanceName;
                localStorage.setItem('selected_instance_alias', instanceName);

                // 🔁 REFRESH COLLECTION LIST IMMEDIATELY
                await this.fetchCollections?.();
                if (!this.collections.includes(this.selected)) {
                    const fallback = this.collections[0] || null;
                    console.warn(`[Auto-select] Old selection '${this.selected}' not in new list. Using '${fallback}'`);
                    this.selected = fallback;
                    localStorage.setItem('selected_collection', fallback);
                }

                // Optionally reload centroid visuals
                await this.loadCentroid();
                await this.fetchWeaviateInstances();


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
                     //   this._confirmUnsavedChanges('Send Message', this._performSendMessage.bind(this));
                     
                            console.log('[Debug] ⚡️ sendMessage() → directly invoking _performSendMessage');
                            await this._performSendMessage();
                          
                    },

                    async _performSendMessage() { // Contains the original message sending logic
                        console.log('[Debug] 🔥 _performSendMessage entered; userInput=', this.userInput);
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
                    
                     
                    // Scroll chat window to bottom after any append
                                
                    scrollToBottom() {
                        this.$nextTick(() => {
                            const win = this.$refs.chatHistoryContainer;
                            if (!win) {
                                console.warn('[scrollToBottom] $refs.chatHistoryContainer missing');
                                return;
                            }
                            win.scrollTop = win.scrollHeight;
                        });
                                },
                    
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
                        if (this.isLoading() || !Array.isArray(this.chatHistory) || this.chatHistory.length === 0) {
                            if (this.showToast) this.showToast("Nothing to save in current chat.", "info");
                            return;
                        }

                        const defaultName = `Chat ${new Date().toLocaleDateString()}`;
                        const chatName = await this.promptForChatName(defaultName);

                        if (!chatName) {
                            if (this.showToast) this.showToast("Save cancelled.", "info");
                            return;
                        }

                        console.log(`[API Call] Saving current chat as: ${chatName}`);
                        this.statusMessage = 'Saving chat...';

                        try {
                            const response = await fetch('/save_chat', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    name: chatName,
                                    history: this.chatHistory
                                })
                            });

                            const result = await response.json();
                            if (!response.ok || !result.success)
                                throw new Error(result.error || `HTTP error! Status: ${response.status}`);

                            this.showToast?.(result.message || `Chat '${chatName}' saved.`, 'success');
                            await this.fetchSavedChats();
                        } catch (error) {
                            console.error('[API Error] Save Chat FAILED:', error);
                            this.statusMessage = 'Save Error';
                            this.showToast?.(`Error saving chat: ${error.message}`, 'error');
                        } finally {
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
                                let newPath = prompt(result.message + "\nEnter new centroid file path (or select):", this.formConfig.paths.CENTROID_DIR);
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
                    

                    // ← Public method that your button should call
                     startIncrementalIngestion() {
                    // Pass both a name and a callback into the confirm helper
                        this._confirmUnsavedChanges(
                                'Incremental Ingestion',
                                () => this._performStartIncrementalIngestion()
                        );
                         },
          
        // 4) Your existing centroid-creation helper
        async _createCentroidFile(centroidPath, dataFolder) {
            try {
                const response = await fetch('/create_centroid', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: new URLSearchParams({
                        centroid_path: centroidPath,
                        data_folder: dataFolder
                    })
                });

                const result = await response.json();

                if (result.status === 'created') {
                    this.showToast?.("Centroid file created successfully. Starting ingestion...", "success");
                    await this._performStartIncrementalIngestion();
                } else {
                    this.showToast?.("Failed to create centroid file: " + (result.message || "Unknown error"), "error");
                }
            } catch (err) {
                console.error("[_createCentroidFile] AJAX error:", err);
                this.showToast?.("AJAX error: " + err.message, "error");
            }
        },
          
            
        // 3) Your existing, battle-tested ingestion logic
        async _performStartIncrementalIngestion() {
            // Prevent double execution
            if (this.isLoading()) {
                console.warn('[Button Click - Core] Incremental Ingestion ignored, already loading:', this.statusMessage);
                return;
            }

            console.log("[API Call] Starting incremental ingestion via /ingest_block...");
            this.statusMessage = 'Starting Incremental Ingestion...';

            // Get config values
            const dataFolder = this.formConfig.paths.DOCUMENT_DIR;
            const centroidPath = this.formConfig.paths.CENTROID_DIR;
            const centroidUpdateMode = this.formConfig.ingestion.CENTROID_UPDATE_MODE; // 'always', 'never', or 'auto'

            try {
                // Build params safely
                const params = new URLSearchParams();
                params.append('data_folder', dataFolder);
                params.append('centroid_path', centroidPath);
                params.append('centroid_update_mode', centroidUpdateMode);
                const rawThreshold = this.formConfig.centroidAutoThreshold;
                const thresholdNum = typeof rawThreshold === 'number'
                    ? rawThreshold
                    : parseFloat(rawThreshold);
                params.append(
                    'centroid_auto_threshold',
                    (Number.isFinite(thresholdNum) ? thresholdNum : 0.5).toString()
                );

                const response = await fetch('/ingest_block', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: params.toString()
                });

                const result = await response.json();

                // Handle centroid missing
                if (result.status === 'centroid_missing') {
                    const newPath = prompt(
                        result.message + "\nEnter new centroid file path (or select):",
                        centroidPath
                    );
                    if (newPath) {
                        await this._createCentroidFile(newPath, dataFolder);
                        await this._performStartIncrementalIngestion();
                    }
                    return;
                }

                // HTTP errors
                if (!response.ok) {
                    const errDetail = result.error || `HTTP error! Status: ${response.status}`;
                    if (result.traceback) console.error("Ingestion Traceback:\n", result.traceback);
                    throw new Error(errDetail);
                }

                // Success: show stats if present
                let successMsg = result.message || 'Incremental ingestion finished.';
                if (result.stats) {
                    const { processed_files, processed_chunks, inserted, errors } = result.stats;
                    const processed = processed_files ?? processed_chunks ?? 'N/A';
                    successMsg += ` (Processed: ${processed}, Inserted: ${inserted ?? 'N/A'}, Errors: ${errors ?? 0})`;
                }
                this.showToast?.(successMsg, 'success', 6000);
                this.statusMessage = 'Ingestion Complete';

            } catch (error) {
                console.error('[API Error] Start Incremental Ingestion FAILED:', error);
                this.statusMessage = 'Ingestion Error';
                this.showToast?.(`Incremental ingestion failed: ${error.message}`, 'error', 10000);

            } finally {
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

                // Keys in env/document that we want to merge rather than overwrite
                const MERGE_ARRAY_KEYS = [
                    'FILE_TYPES',
                    'DOMAIN_KEYWORDS',
                    'AUTO_DOMAIN_KEYWORDS',
                    'USER_ADDED_KEYWORDS'
                ];
                const MERGE_KEYS = [...MERGE_ARRAY_KEYS, 'SELECTED_N_TOP'];

                // 1) Merge top-level sections
                for (const section in loadedConfig) {
                    if (!Object.hasOwn(loadedConfig, section)) continue;
                    const src = loadedConfig[section];
                    const dst = this.formConfig[section];

                    // a) both arrays → replace
                    if (Array.isArray(dst) && Array.isArray(src)) {
                        this.formConfig[section] = [...src];
                        continue;
                    }

                    // b) both plain objects → deep merge
                    if (
                        dst && typeof dst === 'object' && !Array.isArray(dst) &&
                        src && typeof src === 'object' && !Array.isArray(src)
                    ) {
                        this.deepMerge(dst, src);
                        continue;
                    }

                    // c) primitive or type mismatch → warn then assign
                    if (typeof dst !== typeof src) {
                        console.warn(
                            `Section ${section} type mismatch: Frontend=${typeof dst}, Backend=${typeof src}`
                        );
                    }

                    // d) special merge for env/document arrays
                    if (
                        (section === 'env' || section === 'document') &&
                        Object.keys(src).some(key => MERGE_KEYS.includes(key))
                    ) {
                        for (const key in src) {
                            if (MERGE_ARRAY_KEYS.includes(key)) {
                                this.formConfig[section][key] = ensureArray(src[key]);
                            } else {
                                this.formConfig[section][key] = src[key];
                            }
                        }
                    } else {
                        // e) replace whole section
                        this.formConfig[section] = src;
                    }
                }

                // 2) Guarantee critical nested arrays exist
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

                // 3) Populate keyword controls
                const autoKeys = this.formConfig.env.AUTO_DOMAIN_KEYWORDS;
                this.autoDomainKeywordsList = [...autoKeys];
                this.autoDomainKeywords = autoKeys.join(', ');

                // 4) Restore saved Top-N if present
                const savedN = this.formConfig.env.SELECTED_N_TOP;
                if (savedN != null) {
                    this.selectedTopN = savedN;
                }

                // Recompute menu options
                this.allAutoDomainKeywordsList = ensureArray(autoKeys);
                this.updateTopNOptions();
                console.log(`TopN options updated:`, this.topNOptions);

                // 5) Seed the User-Added textarea
                this.envUserKeywordsString = (this.formConfig.env.USER_ADDED_KEYWORDS || []).join(', ');

                // 6) Ensure pipeline.max_history_turns exists and is numeric
                if (!this.formConfig.pipeline) {
                    this.formConfig.pipeline = {};
                }
                if (
                    this.formConfig.pipeline.max_history_turns === undefined ||
                    this.formConfig.pipeline.max_history_turns === null
                ) {
                    console.warn("[loadInitialConfig] max_history_turns missing—defaulting to 5.");
                    this.formConfig.pipeline.max_history_turns = 5;
                }

                // 7) Mark clean
                this._markConfigClean();
                console.log("Config loaded, keywords populated, and state marked clean");

            } catch (error) {
                console.error("Config load failed:", error);
                this.showToast(`Config load error: ${error.message}`, 'error');

                // Reset critical arrays to safe defaults
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
            if (this.isLoadingPresets) {
                console.warn("[UI] Skipping duplicate fetchPresets call.");
                return;
            }

            this.isLoadingPresets = true;
            console.log("[UI] Fetching presets from /list_presets...");

            try {
                const response = await fetch('/list_presets');
                if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);

                this.presets = await response.json();
                this.presetsLoaded = true;

                console.log("[UI] Presets loaded via API:", Object.keys(this.presets).length);
            } catch (error) {
                console.error("[UI] Failed to fetch presets via API:", error);
                this.presets = {};
                this.showToast(`Error loading presets: ${error.message}`, 'error');
            } finally {
                this.isLoadingPresets = false;
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
                                        this.selectedPresetName = '';           // force x-model change
                                        this.presets = { ...this.presets };     // force x-for refresh
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

                    
        fetchCentroidStats() {
            console.warn('[CentroidStats] Fetch triggered. this.selected =', this.selected, 'instance =', this.selectedInstanceAlias);
            if (!this.selected || !this.selectedInstanceAlias) return;

            fetch(`/centroid_stats?collection=${encodeURIComponent(this.selected)}&instance=${encodeURIComponent(this.selectedInstanceAlias)}`)
                .then(res => res.json())
                .then(data => {
                    const meta = data?.stats?.meta;
                    const ok = data?.success && meta && Object.keys(meta).length;

                    if (ok) {
                        console.log("[CentroidStats] Meta received:", meta);
                        this.centroidStats.meta = meta;
                        this.centroidStats.centroid = true;
                        console.log("Final centroidStats.meta", this.centroidStats.meta);
                    } else {
                        console.warn("[CentroidStats] Empty or missing meta payload.");
                        this.centroidStats.meta = {};
                        this.centroidStats.centroid = false;
                    }
                })
                .catch(err => {
                    console.error("[CentroidStats] Failed to fetch:", err);
                    this.centroidStats.meta = {};
                    this.centroidStats.centroid = false;
                });
        },
                    
    
        async fetchAutoDomainKeywords() {
            if (this.isLoadingAutoKeywords || this.autoKeywordsLoaded) {
                console.warn("[UI] Skipping duplicate fetchAutoDomainKeywords call.");
                return;
            }

            this.isLoadingAutoKeywords = true;
            console.log("[UI] Starting fetchAutoDomainKeywords...");

            try {
                const res = await fetch('/get_auto_domain_keywords');
                const data = await res.json();

                this.allAutoDomainKeywordsList = Array.isArray(data.keywords)
                    ? data.keywords
                    : [];

                this.autoDomainKeywords = this.allAutoDomainKeywordsList.length
                    ? this.allAutoDomainKeywordsList.join(', ')
                    : '[No auto domain keywords found]';

                this.activeAutoDomainKeywordsSet = new Set(this.allAutoDomainKeywordsList);

                this.updateTopNOptions();
                this.applyTopNKeywords();

                this.autoKeywordsLoaded = true;

                console.log("[UI] Loaded", this.allAutoDomainKeywordsList.length, "auto keywords.");
            } catch (error) {
                console.error('[UI] Failed to fetch auto domain keywords:', error);
                this.allAutoDomainKeywordsList = [];
                this.autoDomainKeywords = '[Error loading auto domain keywords]';
                this.activeAutoDomainKeywordsSet = new Set();
                this.updateTopNOptions();
            } finally {
                this.isLoadingAutoKeywords = false;
            }
        },
        


            // open the modal, fetch data
            async inspectInstance() {
                try {
                    this.inspectModal.show = true;
                    const res = await fetch('/inspect_instance');
                    if (!res.ok) throw new Error(`HTTP ${res.status}`);
                    this.inspectModal.data = await res.json();
                } catch (err) {
                    console.error("Inspect failed:", err);
                    this.showToast(`Inspect error: ${err.message}`, 'error');
                    this.inspectModal.show = false;
                }
            },
            closeInspectModal() {
                this.inspectModal.show = false;
                this.inspectModal.data = null;
            },
            
            
            // Send the active keywords to backend (update config/file)
          
        
        safeRenderMarkdown(text) {
            if (!text) return '';
            if (!window.marked || !window.DOMPurify) {
                return text
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/\n/g, '<br>');
            }
            // 1) Markdown → HTML
            const rawHtml = window.marked.parse(text);
            // 2) Sanitize allowing only specific attributes
            return DOMPurify.sanitize(rawHtml, {
                ALLOWED_ATTR: ['src', 'class', 'alt', 'title'],
            });
        },
// ─── Markdown + Sanitization Helper ───
safeRenderMarkdown(text) {
    if (!text) return '';
    // If marked or DOMPurify missing, fall back to escaping
    if (!window.marked || !window.DOMPurify) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/\n/g, '<br>');
    }
    // 1) Markdown → HTML
    const rawHtml = window.marked.parse(text);
    // 2) Sanitize allowing images, classes, alt, title
    return DOMPurify.sanitize(rawHtml, {
        ALLOWED_ATTR: ['src', 'class', 'alt', 'title']
    });
},

    }) // <-- End of Alpine.data object
    )
}); // <-- End of event listener


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

// Final log to confirm script parsing
console.log('static/ragapp.js parsed successfully.');

// --- END static/ragapp.js ---//



