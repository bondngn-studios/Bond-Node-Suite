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

/* ---------------------------------------------------------------------------
 * Bond: Big Number 🔢 — large auto-fitting integer display
 * Receives an INT on number_in and renders it huge inside the node body.
 * The digits auto-fit the widget box: drag the node wider to scale up.
 * Color reads from the node's `color` widget (defaults to Bond purple) and
 * updates live without needing a re-run. Displayed value persists on reload.
 * Registered as its own extension so the ShowText handler above is untouched.
 * ------------------------------------------------------------------------- */
app.registerExtension({
    name: "BondNodeSuite.BigNumber",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "BondBigNumber") return;

        const BOND_PURPLE = "#9b30ff";

        function removeNumberWidgets(node) {
            if (!node.widgets) return;
            for (let i = node.widgets.length - 1; i >= 0; i--) {
                if (node.widgets[i].name === "__bond_number__") {
                    node.widgets[i].onRemove?.();
                    node.widgets.splice(i, 1);
                }
            }
        }

        function currentColor(node) {
            const w = node.widgets?.find((x) => x.name === "color");
            const v = (w?.value ?? "").toString().trim();
            return v || BOND_PURPLE;
        }

        function applyColor(node) {
            if (node.__bondNumDiv) node.__bondNumDiv.style.color = currentColor(node);
        }

        // Measure the text at a fixed reference size, then scale the font
        // proportionally so it fills the box on the tighter of the two axes.
        function fitNumber(node) {
            const wrap = node.__bondNumWrap;
            const num  = node.__bondNumDiv;
            if (!wrap || !num) return;
            const w = wrap.clientWidth;
            const h = wrap.clientHeight;
            if (!w || !h) return;
            num.style.fontSize = "100px";
            const tw = num.scrollWidth  || 1;
            const th = num.scrollHeight || 1;
            let fs = 100 * Math.min(w / tw, h / th) * 0.9;
            if (!isFinite(fs) || fs < 8) fs = 8;
            num.style.fontSize = fs + "px";
        }

        function populate(value) {
            removeNumberWidgets(this);

            let raw = Array.isArray(value)
                ? value.flat().find((v) => v !== null && v !== undefined && v !== "")
                : value;
            if (raw === null || raw === undefined) raw = "";
            const text = String(raw);

            const wrap = document.createElement("div");
            wrap.style.cssText = [
                "width:100%",
                "height:100%",
                "display:flex",
                "align-items:center",
                "justify-content:center",
                "overflow:hidden",
                "box-sizing:border-box",
                "padding:4px",
                "background:var(--comfy-input-bg)",
                "border:1px solid var(--border-color)",
                "border-radius:4px",
            ].join(";");

            const num = document.createElement("div");
            num.textContent = text;
            num.style.cssText = [
                "font-family:'Arial Black',Arial,sans-serif",
                "font-weight:900",
                "line-height:1",
                "white-space:nowrap",
                "text-align:center",
                `color:${currentColor(this)}`,
            ].join(";");

            wrap.appendChild(num);
            this.__bondNumWrap = wrap;
            this.__bondNumDiv  = num;

            const node = this;
            this.addDOMWidget("__bond_number__", "customnumber", wrap, {
                getValue() { return num.textContent; },
                setValue(v) {
                    num.textContent = v ?? "";
                    requestAnimationFrame(() => fitNumber(node));
                },
                serialize: true,
                hideOnZoom: false,
                // Height tracks node WIDTH (not height) to avoid a resize
                // feedback loop — dragging the node wider grows the number.
                computeSize(width) {
                    return [width, Math.max(80, Math.round(width * 0.45))];
                },
            });

            applyColor(this);

            if (this.__bondRO) this.__bondRO.disconnect();
            this.__bondRO = new ResizeObserver(() => fitNumber(node));
            this.__bondRO.observe(wrap);

            requestAnimationFrame(() => {
                fitNumber(node);
                app.graph.setDirtyCanvas(true, false);
            });
        }

        // Live color updates without a re-run.
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const cw = this.widgets?.find((w) => w.name === "color");
            if (cw && !cw.__bondHooked) {
                cw.__bondHooked = true;
                const orig = cw.callback;
                const node = this;
                cw.callback = function () {
                    const r = orig?.apply(this, arguments);
                    applyColor(node);
                    return r;
                };
            }
        };

        // Render on execution.
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            populate.call(this, message.number);
        };

        // Persist displayed number across reloads (same approach as Show Text).
        const VALUES = Symbol();
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function () {
            this[VALUES] = arguments[0]?.widgets_values;
            return configure?.apply(this, arguments);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            const wv = this[VALUES];
            if (!wv?.length) return;
            // Scan from the end for the last integer-looking value so the
            // color hex (also a serialized widget) is never shown as the number.
            let found = "";
            for (let i = wv.length - 1; i >= 0; i--) {
                const cand = wv[i] === null || wv[i] === undefined ? "" : String(wv[i]).trim();
                if (/^-?\d+$/.test(cand)) { found = cand; break; }
            }
            if (found) requestAnimationFrame(() => populate.call(this, [found]));
        };
    },
});
