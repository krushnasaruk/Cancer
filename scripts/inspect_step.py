import asyncio, json, websockets, aiohttp

async def test():
    async with aiohttp.ClientSession() as session:
        await session.post('http://127.0.0.1:8000/api/control', json={'action': 'RESET'})
    await asyncio.sleep(0.2)
    async with websockets.connect('ws://127.0.0.1:8000/ws') as ws:
        msg = await ws.recv()
        d = json.loads(msg)
        print('=== AFTER RESET ===')
        for jname, info in d['joints'].items():
            print(f"  {jname}: angle={info['angle_deg']:.1f} deg, target={info['target_deg']:.1f} deg")
        print(f"Base z={d['base']['z']*1000:.1f}mm, roll={d['base']['roll']:.1f}, pitch={d['base']['pitch']:.1f}")

    async with aiohttp.ClientSession() as session:
        await session.post('http://127.0.0.1:8000/api/control', json={'action': 'START'})
    
    await asyncio.sleep(0.1)
    async with websockets.connect('ws://127.0.0.1:8000/ws') as ws:
        for i in range(10):
            msg = await ws.recv()
            d = json.loads(msg)
            print(f"--- Step {i} ---")
            print(f"Base z={d['base']['z']*1000:.1f}mm, roll={d['base']['roll']:.1f}, pitch={d['base']['pitch']:.1f}")
            for jname, info in d['joints'].items():
                print(f"  {jname}: actual={info['angle_deg']:.1f}deg target={info['target_deg']:.1f}deg torque={info['torque_nm']:.3f}")

asyncio.run(test())
