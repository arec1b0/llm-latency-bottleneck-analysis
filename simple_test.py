import asyncio
import aiohttp

async def test_health():
    async with aiohttp.ClientSession() as session:
        async with session.get('http://localhost:8000/health') as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f'Health: {data["status"]}')
                print(f'Model loaded: {data["model_loaded"]}')
                return data
            else:
                print(f'Error: {resp.status}')
                return None

if __name__ == '__main__':
    asyncio.run(test_health())
