import asyncio, json, websockets, aiohttp

async def test_modes():
    modes = ["STAND", "WALK", "SINE", "WAVE", "PUSHUP"]
    async with aiohttp.ClientSession() as session:
        for m in modes:
            await session.post('http://127.0.0.1:8000/api/control', json={'action': 'RESET'})
            res = await session.post('http://127.0.0.1:8000/api/control', json={'action': 'SET_PID_MODE', 'mode': m})
            await session.post('http://127.0.0.1:8000/api/control', json={'action': 'START'})
            print(f"Testing mode {m}: status={res.status}")
            await asyncio.sleep(0.3)
            
            async with websockets.connect('ws://127.0.0.1:8000/ws') as ws:
                msg = await ws.recv()
                d = json.loads(msg)
                bz = d['base']['z']
                roll = d['base']['roll']
                pitch = d['base']['pitch']
                print(f"  -> Mode {m}: Z={bz*1000:.1f}mm, Roll={roll:.1f}°, Pitch={pitch:.1f}°")

asyncio.run(test_modes())
