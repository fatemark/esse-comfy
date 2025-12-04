import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

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
