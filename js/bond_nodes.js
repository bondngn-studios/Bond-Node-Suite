/**
 * Bond Node Suite — Frontend Extension
 * Handles inline text rendering for Bond: Show Text node.
 * Sets default colors for Bond metadata nodes.
 * Uses addDOMWidget (no deprecated imports) with proper sizing.
 */
import { app } from "../../scripts/app.js";

// Default purple color for all Bond metadata nodes
const METADATA_NODES = new Set([
    "BondSaveWithCustomMetadata",
    "BondSaveVideoWithMetadata",
    "BondReadMetadata",
    "BondStripMetadata",
    "BondGlobalMetadataSettings",
]);

const METADATA_COLOR   = "#1e0231";
const METADATA_BGCOLOR = "#16011f";

app.registerExtension({
    name: "BondNodeSuite.ShowText",

    // nodeCreated fires after full instantiation — correct place to set colors
    // comfyClass is the reliable identifier matching NODE_CLASS_MAPPINGS keys
    nodeCreated(node) {
        const id = node.comfyClass ?? node.type;
        if (METADATA_NODES.has(id)) {
            node.color   = METADATA_COLOR;
            node.bgcolor = METADATA_BGCOLOR;
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "BondShowText") return;

        function removeTextWidgets(node) {
            if (!node.widgets) return;
            for (let i = node.widgets.length - 1; i >= 0; i--) {
                if (node.widgets[i].name === "__bond_text__") {
                    node.widgets[i].onRemove?.();
                    node.widgets.splice(i, 1);
                }
            }
        }

        function populate(text) {
            removeTextWidgets(this);

            const v = Array.isArray(text) ? text : [text];
            const value = v.flat().filter(Boolean).join("\n");
            if (!value.trim()) return;

            const textarea = document.createElement("textarea");
            textarea.readOnly = true;
            textarea.value    = value;
            textarea.style.cssText = [
                "width:100%",
                "font-size:11px",
                "font-family:monospace",
                "background:var(--comfy-input-bg)",
                "color:var(--input-text)",
                "border:1px solid var(--border-color)",
                "padding:4px",
                "box-sizing:border-box",
                "resize:none",
                `background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='80'%3E%3Ctext x='50%25' y='55%25' font-family='Arial Black,sans-serif' font-size='52' font-weight='900' fill='%23ffffff' fill-opacity='0.03' text-anchor='middle' dominant-baseline='middle' letter-spacing='8'%3EBOND%3C/text%3E%3C/svg%3E")`,
                "background-repeat:no-repeat",
                "background-position:center center",
                "background-size:cover",
            ].join(";");

            this.addDOMWidget("__bond_text__", "customtext", textarea, {
                getValue() { return textarea.value; },
                setValue(v) { textarea.value = v || ""; },
                serialize: true,
                computeSize(width) {
                    const lines = textarea.value.split("\n").length;
                    const h     = Math.max(60, lines * 15 + 12);
                    return [width, h];
                },
            });

            requestAnimationFrame(() => {
                const sz = this.computeSize();
                if (sz[0] < this.size[0]) sz[0] = this.size[0];
                if (sz[1] < this.size[1]) sz[1] = this.size[1];
                this.onResize?.(sz);
                app.graph.setDirtyCanvas(true, false);
            });
        }

        // Render on execution
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            onExecuted?.apply(this, arguments);
            populate.call(this, message.text);
        };

        // Store widget values before configure strips them
        const VALUES = Symbol();
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function() {
            this[VALUES] = arguments[0]?.widgets_values;
            return configure?.apply(this, arguments);
        };

        // Restore persisted text on page reload
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            onConfigure?.apply(this, arguments);
            const wv = this[VALUES];
            if (wv?.length) {
                requestAnimationFrame(() => {
                    const saved = wv[wv.length - 1];
                    if (saved && typeof saved === "string" && saved.trim()) {
                        populate.call(this, [saved]);
                    }
                });
            }
        };
    },
});
