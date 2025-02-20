from tetra.client import remote
import asyncio




@remote("server1")
def say_hello(name: str) -> str:
    return f"Hello, {name}!"



async def main():
    result = await say_hello("Alice")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())



