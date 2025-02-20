import runpod
import asyncio


def deploy_endpoint(config, type) -> None:
    runpod.api_key = ""
    try:
        new_endpoint = runpod.create_endpoint(
        name="test",
        template_id="ib4coc7w60",
        gpu_ids="AMPERE_16",
        workers_min=0,
        workers_max=1,
        )
        # Output the created endpoint details
        print(f"endpoint crearted: {new_endpoint}")
        return f"https://api.runpod.ai/v2/{new_endpoint['id']}/"
    except Exception as e:
        raise e
    


async def provision_resource():
    from tetra.client_manager import get_global_client
    client = get_global_client()

    endpoint_configs = [{
        "name": "test_endpoint",
        "template": "Pre_built",
        "gpu_ids": ["0"],
        "env_vars": { "API_KEY": "test_key"} 
    },{
        "name": "test_endpoint",
        "template": "Pre_built",
        "gpu_ids": ["0"],
        "env_vars": { "API_KEY": "test_key"} 
    }]

    for config in endpoint_configs:
        await client.add_server("config.name", deploy_endpoint(config))  # Here we need to provide the server name and the function that we want to run on that server and this will be IP.
        print(f"This is the pool, {client.get_server('config.name')}")




