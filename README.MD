***New for this branch so far***
Summary Table of Key Integration Points
Layer	File(s)	Function/Class/Method	Purpose/Action
Backend Logic	centroid_manager.py	CentroidManager	Centralizes all centroid operations
API	app.py	/api/centroid, /api/centroid/query_insight	Expose centroid and query feedback endpoints
Pipeline	pipeline.py	generate_response	Attach centroid feedback to every query
Ingestion	ingest_block.py, ingest_docs_v7.py	after_ingestion	Recalculate and save centroid post-ingestion
Frontend State	ragapp.js	centroidStats, queryCentroidInsight	Store and fetch centroid data and insights
UI	index.html	centroid-panel	Display centroid stats and per-query feedback

