"""Debug: Start simulation and check if robot stays standing + test walk gait."""
import json, asyncio, websockets, aiohttp

async def test():
    # Start the simulation
    async with aiohttp.ClientSession() as session:
        await session.post("http://127.0.0.1:8000/api/control", json={"action": "START"})
        print("Sent START command")
    
    await asyncio.sleep(0.5)  # Let physics run a bit
    
    async with websockets.connect("ws://127.0.0.1:8000/ws") as ws:
        for i in range(5):
            msg = await ws.recv()
            d = json.loads(msg)
            bz = d["base"]["z"]
            roll = d["base"]["roll"]
            pitch = d["base"]["pitch"]
            step = d["metrics"]["steps"]
            
            joints_ok = all(
                abs(jv["angle_deg"] - jv["target_deg"]) < 5
                for jv in d["joints"].values()
            )
            
            status = "STANDING" if bz > 0.06 and abs(roll) < 10 and abs(pitch) < 10 else "FALLEN"
            print(f"Frame {i}: Z={bz*1000:.1f}mm Roll={roll:.1f}° Pitch={pitch:.1f}° Step={step} JointsOK={joints_ok} -> {status}")
    
    # Test walk gait
    async with aiohttp.ClientSession() as session:
        await session.post("http://127.0.0.1:8000/api/control", json={"action": "SET_PID_MODE", "mode": "WALK"})
        print("\nSent WALK gait command")
    
    await asyncio.sleep(1.0)
    
    async with websockets.connect("ws://127.0.0.1:8000/ws") as ws:
        for i in range(5):
            msg = await ws.recv()
            d = json.loads(msg)
            bz = d["base"]["z"]
            bx = d["base"]["x"]
            roll = d["base"]["roll"]
            pitch = d["base"]["pitch"]
            step = d["metrics"]["steps"]
            status = "WALKING" if bz > 0.04 and abs(roll) < 20 and abs(pitch) < 20 else "FALLEN"
            print(f"Walk {i}: X={bx*1000:.1f}mm Z={bz*1000:.1f}mm Roll={roll:.1f}° Pitch={pitch:.1f}° Step={step} -> {status}")

asyncio.run(test())
