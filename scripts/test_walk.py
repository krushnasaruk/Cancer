import asyncio, json, websockets, aiohttp

async def test_walk():
    async with aiohttp.ClientSession() as session:
        await session.post('http://127.0.0.1:8000/api/control', json={'action': 'RESET'})
        await session.post('http://127.0.0.1:8000/api/control', json={'action': 'SET_PID_MODE', 'mode': 'WALK'})
        await session.post('http://127.0.0.1:8000/api/control', json={'action': 'START'})
    
    print('Started WALK mode...')
    await asyncio.sleep(0.1)
    
    async with websockets.connect('ws://127.0.0.1:8000/ws') as ws:
        for i in range(25):
            msg = await ws.recv()
            d = json.loads(msg)
            bx = d['base']['x']
            by = d['base']['y']
            bz = d['base']['z']
            roll = d['base']['roll']
            pitch = d['base']['pitch']
            print(f"Step {i:02d}: pos=({bx*1000:6.1f}, {by*1000:5.1f}, {bz*1000:5.1f}) mm | roll={roll:5.1f}° pitch={pitch:5.1f}° | contacts FL={d['contacts']['FL']} FR={d['contacts']['FR']} RL={d['contacts']['RL']} RR={d['contacts']['RR']}")

asyncio.run(test_walk())
