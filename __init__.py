from .esse_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .utils import PresetManager
from aiohttp import web
import server

# Initialize PresetManager
manager = PresetManager()

# Setup API Routes
@server.PromptServer.instance.routes.get("/esse/presets")
async def get_presets(request):
    return web.json_response(manager.load_presets())

@server.PromptServer.instance.routes.post("/esse/presets/save")
async def save_preset(request):
    data = await request.json()
    category = data.get("category")
    key = data.get("key")
    value = data.get("value")

    if category and key:
        manager.save_preset(category, key, value)
        return web.json_response({"status": "success"})
    return web.json_response({"status": "error", "message": "Missing category or key"}, status=400)

@server.PromptServer.instance.routes.post("/esse/presets/category/add")
async def add_category(request):
    data = await request.json()
    category = data.get("category")

    if category:
        if manager.add_category(category):
            return web.json_response({"status": "success"})
        else:
            return web.json_response({"status": "exists", "message": "Category already exists"})
    return web.json_response({"status": "error", "message": "Missing category"}, status=400)

@server.PromptServer.instance.routes.post("/esse/presets/delete")
async def delete_item(request):
    data = await request.json()
    category = data.get("category")
    key = data.get("key") # Optional, if missing delete category?

    if category:
        if key:
            manager.delete_preset(category, key)
        else:
            manager.delete_category(category)
        return web.json_response({"status": "success"})
    return web.json_response({"status": "error", "message": "Missing category"}, status=400)

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
