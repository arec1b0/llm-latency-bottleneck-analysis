import asyncio
import aiohttp
import json

async def test_api():
    "
Simple
API
functionality
test."
    base_url = http://localhost:8000
    
    print(Testing
API
functionality...)
    
    async with aiohttp.ClientSession() as session:
        # Test health endpoint
        print(1.
Testing
/health
endpoint...)
        async with session.get(f
base_url
/health) as resp:
            if resp.status == 200:
                health = await resp.json()
                print(f
Status:
health['status']
)
                print(f
Model
loaded:
health['model_loaded']
)
                print(f
GPU
available:
health['gpu_available']
)
            else:
                print(f
ERROR:
Health
check
failed
with
status
resp.status
)
                return False
        
        # Load model if not loaded
        if not health.get('model_loaded'):
            print(
2.
Loading
model...)
            async with session.post(f
base_url
/model/load) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f
Model
load
result:
result['message']
)
                else:
                    print(f
ERROR:
Model
load
failed
with
status
resp.status
)
                    return False
        
        # Test generation endpoint
        print(
3.
Testing
/generate
endpoint...)
        request_data = {
            prompt: Hello
world,
            max_tokens: 10,
            temperature: 0.7,
            top_p: 0.9,
            do_sample: True
        }
        
        async with session.post(f
base_url
/generate, json=request_data) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f
Generated
text:
result['generated_text'][:50]
...)
                print(f
TTFT:
result['ttft']:.3f
s)
                print(f
TPOT:
result['tpot']:.3f
s)
                print(f
Tokens:
result['total_tokens']
)
                return True
            else:
                error = await resp.text()
                print(f
ERROR:
Generation
failed
with
status
resp.status
)
                print(f
Error:
error
)
                return False

if __name__ == 
__main__:
    asyncio.run(test_api())
