import asyncio
import aiohttp
import json

async def test_generation():
    async with aiohttp.ClientSession() as session:
        # Test generation
        request_data = {
            'prompt': 'Hello world',
            'max_tokens': 10,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True
        }
        
        print('Testing generation...')
        async with session.post('http://localhost:8000/generate', json=request_data) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f'Generated: {result["generated_text"]}')
                print(f'TTFT: {result["ttft"]:.3f}s')
                print(f'TPOT: {result["tpot"]:.3f}s')
                print(f'Tokens: {result["total_tokens"]}')
                return True
            else:
                error = await resp.text()
                print(f'Generation failed: {resp.status} - {error}')
                return False

if __name__ == '__main__':
    result = asyncio.run(test_generation())
    print(f'Test result: {"PASSED" if result else "FAILED"}')
