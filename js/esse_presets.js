import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ---------------------------------------------------------
// SCROLLABLE DROPDOWN
// LiteGraph draws the combo menu at its full height and scales it
// with the canvas, so a long list of categories/keys runs off the
// screen with no way to reach the entries at the bottom.
// The combos below use this dropdown instead: it is clamped to the
// visible area, scrolls, and filters once the list gets long.
// ---------------------------------------------------------

const DROPDOWN_STYLE_ID = "esse-dropdown-styles";
const FILTER_THRESHOLD = 10;
const WIDGET_HEIGHT = 20;

function injectDropdownStyles() {
    if (document.getElementById(DROPDOWN_STYLE_ID)) return;

    const style = document.createElement("style");
    style.id = DROPDOWN_STYLE_ID;
    style.textContent = `
.esse-dropdown {
    position: fixed;
    top: 0;
    left: -9999px;
    z-index: 10000;
    display: flex;
    flex-direction: column;
    min-width: 160px;
    max-width: 480px;
    padding: 4px;
    overflow: hidden;
    border: 1px solid var(--border-default, #4e4e4e);
    border-radius: 6px;
    background-color: var(--comfy-menu-bg, #353535);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
    font-family: Arial, sans-serif;
    font-size: 12px;
}
.esse-dropdown-filter {
    flex: 0 0 auto;
    margin: 2px 2px 4px;
    padding: 4px 6px;
    border: 1px solid var(--border-default, #4e4e4e);
    border-radius: 4px;
    outline: none;
    background-color: var(--comfy-input-bg, #222);
    color: var(--input-text, #ddd);
    font-size: 12px;
}
.esse-dropdown-list {
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
    overscroll-behavior: contain;
    scrollbar-width: thin;
}
.esse-dropdown-list::-webkit-scrollbar {
    width: 8px;
}
.esse-dropdown-list::-webkit-scrollbar-thumb {
    border-radius: 4px;
    background-color: var(--border-default, #4e4e4e);
}
.esse-dropdown-item {
    padding: 4px 8px;
    border-radius: 4px;
    color: var(--input-text, #aaa);
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.esse-dropdown-item.selected {
    font-weight: bold;
}
.esse-dropdown-item.active {
    background-color: var(--palette-interface-panel-hover-surface, rgba(255, 255, 255, 0.1));
    color: var(--content-hover-fg, #fff);
}
.esse-dropdown-empty {
    padding: 4px 8px;
    color: #888;
    font-style: italic;
}
`;
    document.head.appendChild(style);
}

let openDropdownInstance = null;

function closeDropdown() {
    openDropdownInstance?.close();
}

// Places the dropdown next to the widget, flipping above it when there is
// more room there, and caps its height to whatever space is left on screen.
function positionDropdown(root, anchor) {
    const margin = 8;
    const viewWidth = window.innerWidth;
    const viewHeight = window.innerHeight;

    const spaceBelow = viewHeight - anchor.bottom - margin;
    const spaceAbove = anchor.top - margin;
    const flip = spaceBelow < 160 && spaceAbove > spaceBelow && spaceAbove >= 120;

    root.style.maxHeight = `${Math.max(120, Math.min(viewHeight * 0.7, flip ? spaceAbove : spaceBelow))}px`;
    if (anchor.width) {
        root.style.minWidth = `${Math.min(Math.max(anchor.width, 160), 480)}px`;
    }

    root.style.left = `${Math.max(margin, Math.min(anchor.left, viewWidth - root.offsetWidth - margin))}px`;

    if (flip) {
        // Anchored from the bottom so the list stays against the widget while filtering.
        root.style.top = "auto";
        root.style.bottom = `${viewHeight - anchor.top + 2}px`;
    } else {
        root.style.bottom = "auto";
        root.style.top = `${Math.max(margin, Math.min(anchor.bottom + 2, viewHeight - root.offsetHeight - margin))}px`;
    }
}

function openDropdown({ values, current, anchor, onSelect }) {
    injectDropdownStyles();
    closeDropdown();

    const previousFocus = document.activeElement;

    const root = document.createElement("div");
    root.className = "esse-dropdown";

    let filter = null;
    if (values.length > FILTER_THRESHOLD) {
        filter = document.createElement("input");
        filter.className = "esse-dropdown-filter";
        filter.placeholder = "Filter...";
        filter.autocomplete = "off";
        filter.spellcheck = false;
        root.appendChild(filter);
    }

    const list = document.createElement("div");
    list.className = "esse-dropdown-list";
    root.appendChild(list);

    const items = values.map((value) => {
        const el = document.createElement("div");
        el.className = "esse-dropdown-item";
        el.textContent = String(value);
        el.title = String(value);
        if (value === current) el.classList.add("selected");
        list.appendChild(el);
        return { value, el, visible: true };
    });

    const empty = document.createElement("div");
    empty.className = "esse-dropdown-empty";
    empty.textContent = "No matches";
    empty.style.display = "none";
    list.appendChild(empty);

    let activeIndex = -1;

    function setActive(index, scroll) {
        if (activeIndex >= 0) items[activeIndex].el.classList.remove("active");
        activeIndex = index;
        if (index < 0) return;
        items[index].el.classList.add("active");
        if (scroll) items[index].el.scrollIntoView({ block: "nearest" });
    }

    function move(delta) {
        const visible = items.map((item, index) => ({ item, index })).filter((e) => e.item.visible);
        if (!visible.length) return;
        const current = visible.findIndex((e) => e.index === activeIndex);
        const next = current < 0
            ? (delta > 0 ? 0 : visible.length - 1)
            : (current + delta + visible.length) % visible.length;
        setActive(visible[next].index, true);
    }

    function applyFilter() {
        const query = (filter?.value || "").toLowerCase();
        let matches = 0;
        for (const item of items) {
            item.visible = !query || String(item.value).toLowerCase().includes(query);
            item.el.style.display = item.visible ? "" : "none";
            if (item.visible) matches++;
        }
        empty.style.display = matches ? "none" : "";
        if (activeIndex >= 0 && !items[activeIndex].visible) setActive(-1);
        if (activeIndex < 0) move(1);
    }

    function close() {
        if (openDropdownInstance !== instance) return;
        openDropdownInstance = null;
        document.removeEventListener("pointerdown", onPointerDown, true);
        document.removeEventListener("keydown", onKeyDown, true);
        window.removeEventListener("wheel", onWheel, true);
        window.removeEventListener("resize", close);
        window.removeEventListener("blur", close);
        root.remove();
        if (previousFocus?.isConnected) previousFocus.focus?.();
    }

    function choose(value) {
        close();
        onSelect(value);
    }

    function onPointerDown(e) {
        if (!root.contains(e.target)) close();
    }

    // Scrolling the canvas would leave the dropdown floating next to nothing.
    function onWheel(e) {
        if (!root.contains(e.target)) close();
    }

    function onKeyDown(e) {
        switch (e.key) {
            case "ArrowDown":
                move(1);
                break;
            case "ArrowUp":
                move(-1);
                break;
            case "Enter":
                if (activeIndex < 0) return;
                choose(items[activeIndex].value);
                break;
            case "Escape":
                close();
                break;
            default:
                return;
        }
        e.preventDefault();
        e.stopPropagation();
    }

    items.forEach((item, index) => {
        item.el.addEventListener("pointerenter", () => setActive(index));
        item.el.addEventListener("click", () => choose(item.value));
    });
    filter?.addEventListener("input", applyFilter);

    document.body.appendChild(root);
    positionDropdown(root, anchor);

    const currentIndex = items.findIndex((item) => item.value === current);
    setActive(currentIndex < 0 ? 0 : currentIndex, true);
    filter?.focus();

    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("wheel", onWheel, true);
    window.addEventListener("resize", close);
    window.addEventListener("blur", close);

    const instance = { close };
    openDropdownInstance = instance;
    return instance;
}

// Screen rect of the widget, so the dropdown opens under it like a
// native select. Falls back to the cursor if the maths look wrong.
function widgetAnchor(node, widget, canvas, e) {
    const cursor = { left: e.clientX, top: e.clientY - 8, bottom: e.clientY + 8, width: 0 };

    const element = canvas?.canvas;
    const ds = canvas?.ds;
    if (!element || !ds) return cursor;

    const scale = ds.scale || 1;
    const offset = ds.offset || [0, 0];
    const rect = element.getBoundingClientRect();
    const left = rect.left + (node.pos[0] + 15 + offset[0]) * scale;
    const widgetY = widget.last_y ?? widget.y ?? 0;
    const top = rect.top + (node.pos[1] + widgetY + offset[1]) * scale;
    const width = ((widget.width || node.size[0]) - 30) * scale;
    const height = WIDGET_HEIGHT * scale;

    if (![left, top, width, height].every(Number.isFinite)) return cursor;
    if (top + height < 0 || top > window.innerHeight || left > window.innerWidth) return cursor;

    return { left, top, bottom: top + height, width };
}

// Replaces the combo's built-in menu, keeping the left/right stepper arrows.
function enableScrollableDropdown(widget) {
    widget.onClick = function ({ e, node, canvas }) {
        const width = this.width || node.size[0];
        const x = (e.canvasX ?? 0) - node.pos[0];
        if (e.canvasX !== undefined) {
            if (x < 40) return this.decrementValue?.({ e, node, canvas });
            if (x > width - 40) return this.incrementValue?.({ e, node, canvas });
        }

        const raw = typeof this.options?.values === "function"
            ? this.options.values(this, node)
            : this.options?.values;
        const values = Array.isArray(raw) ? raw : Object.keys(raw || {});
        if (!values.length) return;

        openDropdown({
            values,
            current: this.value,
            anchor: widgetAnchor(node, this, canvas, e),
            onSelect: (value) => {
                if (value !== this.value && typeof this.setValue === "function") {
                    this.setValue(value, { e, node, canvas });
                } else {
                    // Re-picking the current entry ("+ Add New +") still has to act.
                    this.value = value;
                    this.callback?.(value, canvas, node, canvas?.graph_mouse, e);
                }
                canvas?.setDirty?.(true, true);
            },
        });
    };
}

app.registerExtension({
    name: "Esse.PromptingPresets",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "EssePromptingPresets") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const node = this;
                node.presets = {};

                // Find original widgets
                // We assume order: category, key, value based on INPUT_TYPES in python
                // or we can search by name if they are named.
                // ComfyUI usually names them.
                const widgetCategory = node.widgets.find(w => w.name === "category");
                const widgetKey = node.widgets.find(w => w.name === "key");
                const widgetValue = node.widgets.find(w => w.name === "value");

                if (!widgetCategory || !widgetKey || !widgetValue) {
                    console.error("EssePromptingPresets: Could not find required widgets.");
                    return;
                }

                // Helper to create a dropdown
                function createCombo(name, options, callback) {
                    const combo = node.addWidget("combo", name, options[0], callback, { values: options });
                    return combo;
                }

                // ---------------------------------------------------------
                // DATA HANDLING
                // ---------------------------------------------------------

                node.fetchPresets = async () => {
                    try {
                        const response = await api.fetchApi("/esse/presets");
                        if (response.status === 200) {
                            node.presets = await response.json();
                            node.updateCategoryOptions();
                        }
                    } catch (e) {
                        console.error("Error fetching presets:", e);
                    }
                };

                node.savePreset = async () => {
                    const cat = widgetCategory.value;
                    const key = widgetKey.value;
                    const val = widgetValue.value;

                    if (!cat || !key) {
                        alert("Category and Key are required.");
                        return;
                    }

                    try {
                        const response = await api.fetchApi("/esse/presets/save", {
                            method: "POST",
                            body: JSON.stringify({ category: cat, key: key, value: val }),
                            headers: { "Content-Type": "application/json" }
                        });

                        if (response.status === 200) {
                            // Reload presets to ensure consistency
                            await node.fetchPresets();
                            // Re-select current to restore state
                            // node.categoryCombo.value = cat; // Should persist
                            // node.updateKeyOptions(cat);
                        } else {
                            alert("Failed to save preset.");
                        }
                    } catch (e) {
                        console.error("Error saving preset:", e);
                        alert("Error saving preset.");
                    }
                };

                node.addCategory = async (name) => {
                    try {
                        const response = await api.fetchApi("/esse/presets/category/add", {
                            method: "POST",
                            body: JSON.stringify({ category: name }),
                            headers: { "Content-Type": "application/json" }
                        });
                         if (response.status === 200) {
                            await node.fetchPresets();
                            // Select the new category
                            if (node.categoryCombo) {
                                node.categoryCombo.value = name;
                                widgetCategory.value = name;
                                node.updateKeyOptions(name);
                            }
                        }
                    } catch (e) {
                         console.error("Error adding category:", e);
                    }
                };


                // ---------------------------------------------------------
                // UI SETUP
                // ---------------------------------------------------------

                // 1. Category Combo
                // We create a new combo widget and hide the original text widget (or keep it sync'd)
                // Actually, let's keep the original text widget but maybe hide it?
                // Or easier: use the text widget as the "value holder" and use the combo to set it.
                // But `widgetCategory` IS the input to the node.

                // Let's create a Combo widget *above* the category text widget.
                // And maybe another Combo for Keys *above* the key text widget.

                // Remove existing widgets temporarily to re-add in order?
                // LiteGraph doesn't make reordering easy.
                // But we can `splice` into `node.widgets`.

                // Helper: Create Combo for Category
                const categoryOptions = ["Loading..."];
                node.categoryCombo = node.addWidget("combo", "Select Category", categoryOptions[0], (value) => {
                    if (value === "+ Add New +") {
                        const name = prompt("Enter new category name:");
                        if (name) {
                            node.addCategory(name);
                        } else {
                            // Revert to previous or default
                            node.categoryCombo.value = widgetCategory.value || Object.keys(node.presets)[0];
                        }
                    } else {
                        widgetCategory.value = value;
                        // Force visual update if inputEl exists (standard Comfy/LiteGraph widgets often use this)
                        if (widgetCategory.inputEl) {
                            widgetCategory.inputEl.value = value;
                        }
                        node.updateKeyOptions(value);
                    }
                }, { values: categoryOptions });

                // Helper: Create Combo for Key
                const keyOptions = ["Select a Category first"];
                node.keyCombo = node.addWidget("combo", "Select Key", keyOptions[0], (value) => {
                    if (value === "+ Add New +") {
                        // Just clear the Key field so user can type
                         widgetKey.value = "New Key";
                         widgetValue.value = "";
                    } else if (value && value !== "Select a Category first") {
                        widgetKey.value = value;
                        if (widgetKey.inputEl) {
                            widgetKey.inputEl.value = value;
                        }
                        const cat = node.categoryCombo.value;
                        if (cat && node.presets[cat]) {
                            const val = node.presets[cat][value] || "";
                            widgetValue.value = val;
                            if (widgetValue.inputEl) {
                                widgetValue.inputEl.value = val;
                            }
                        }
                    }
                }, { values: keyOptions });

                // Preset lists get long, so both selects use a dropdown that scrolls.
                enableScrollableDropdown(node.categoryCombo);
                enableScrollableDropdown(node.keyCombo);

                // Add Save Button
                node.addWidget("button", "Save Preset", null, () => {
                    node.savePreset();
                });


                // ---------------------------------------------------------
                // WIDGET MANAGEMENT
                // ---------------------------------------------------------

                // We want the layout:
                // 1. Select Category (Combo)
                // 2. Category Name (Text) - widgetCategory
                // 3. Select Key (Combo)
                // 4. Key Name (Text) - widgetKey
                // 5. Value (Text) - widgetValue
                // 6. Save Button

                // Current Order in node.widgets:
                // [category, key, value, categoryCombo, keyCombo, saveButton]

                // Let's reorder them.
                const newWidgets = [
                    node.categoryCombo,
                    widgetCategory,
                    node.keyCombo,
                    widgetKey,
                    widgetValue,
                    node.widgets[node.widgets.length - 1] // Save Button
                ];

                node.widgets = newWidgets;


                // ---------------------------------------------------------
                // UPDATE LOGIC
                // ---------------------------------------------------------

                node.updateCategoryOptions = () => {
                    const categories = Object.keys(node.presets);
                    const options = [...categories, "+ Add New +"];
                    node.categoryCombo.options.values = options;

                    // If current text widget value is valid, select it
                    if (widgetCategory.value && categories.includes(widgetCategory.value)) {
                        node.categoryCombo.value = widgetCategory.value;
                        node.updateKeyOptions(widgetCategory.value);
                    } else if (categories.length > 0) {
                        // Default to first
                        node.categoryCombo.value = categories[0];
                        widgetCategory.value = categories[0];
                        node.updateKeyOptions(categories[0]);
                    }
                };

                node.updateKeyOptions = (category) => {
                    if (!category || !node.presets[category]) {
                        node.keyCombo.options.values = ["Select a Category first"];
                        node.keyCombo.value = "Select a Category first";
                        return;
                    }

                    const keys = Object.keys(node.presets[category]);
                    const options = [...keys, "+ Add New +"];
                    node.keyCombo.options.values = options;

                    // Try to match current key
                    if (widgetKey.value && keys.includes(widgetKey.value)) {
                        node.keyCombo.value = widgetKey.value;
                        // Update value too? Maybe not if user was editing?
                        // Only update value if we just switched categories?
                        // For now, let's sync value from preset if key matches.
                        widgetValue.value = node.presets[category][widgetKey.value];
                    } else if (keys.length > 0) {
                        node.keyCombo.value = keys[0];
                        widgetKey.value = keys[0];
                        widgetValue.value = node.presets[category][keys[0]];
                    } else {
                         // Empty category
                         node.keyCombo.value = "+ Add New +";
                         widgetKey.value = "New Key";
                         widgetValue.value = "";
                    }
                };

                // Initial Fetch
                node.fetchPresets();
            };
        }
    }
});
