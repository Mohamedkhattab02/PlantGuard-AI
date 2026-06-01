"""Service layer for PlantGuard AI.

Each module owns one concern and is independently importable and testable:

- :mod:`store`        — Firebase Realtime DB wrapper (graceful degradation)
- :mod:`gamification` — per-user, persistent mission/point engine
- :mod:`documents`    — PDF corpus loading (parsed once)
- :mod:`rag`          — semantic retrieval + Gemini generation
- :mod:`image`        — disease classifier, label parsing, validation
- :mod:`iot`          — sensor history, concurrent fetch, MapReduce
- :mod:`alerts`       — agronomic threshold rules
- :mod:`history`      — per-user diagnosis history
- :mod:`weather`      — weather context (Open-Meteo, no key required)
- :mod:`reports`      — PDF diagnosis reports
"""
