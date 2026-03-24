# ═══════════════════════════════════════════════════════════════════════════
# Backend/api/main.py — Portfolio Simulator
# ═══════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from api.routers import operations, assets, results, portfolio
from api.deps import get_db
#import asyncio

app = FastAPI(
    title="Backtesting Platform API",
    description="Backend API for Backtesting Platform",
    version="0.1.0"
)

# Cors - Allows React frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allowed_origins=["http://localhost:5173"], # Vite dev server
    allowed_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registers routers
app.include_router(operations.router)
app.include_router(assets.router)
app.include_router(results.router)
app.include_router(portfolio.router)

@app.get("/health")
def health(): # Health check, confirms that the API is running
    return {"status": "ok"}

@app.on_event("startup")
def startup(): # Initiates database connection on startup
    get_db()
    print("     > [API] Database connected.")

# ── WebSocket — backtest progress streaming ────────────────────────
activate_connections: list[WebSocket]=[]

@app.websocket("/ws/backtest")
async def websocket_backtest(websocket: WebSocket):
    # WebSocket for real-time backtest progress streaming
    # Frontend connects here to receive live updates

    await websocket.accept()
    activate_connections.append(websocket)

    try:
        while True: # Waits for client message (keepalive or future commands)
            data = await websocket.receive_text()
            await websocket.send_json({"type": "ack", "message": data})
    except WebSocketDisconnect:
        activate_connections.remove(websocket)

async def broadcast(message: dict):
    # Broadcasts a message to all connected WebSocket clients
    # Called by Operation.run() to stream progress to the frontend

    for connection in activate_connections:
        try:
            await connection.send_json(message)
        except Exception:
            activate_connections.remove(connection)

# ── Run directly ──────────────────────────────────────────────────────────
# uvicorn api.main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)



















