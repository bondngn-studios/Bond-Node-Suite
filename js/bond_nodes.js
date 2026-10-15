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
 *
 * Drawn directly on the canvas (no DOM element, no ResizeObserver): every
 * frame the digits are measured with ctx.measureText and scaled to fit the
 * space below the widgets, so the number always fits whatever size the node
 * is dragged to — width or height — and scales cleanly with canvas zoom.
 *
 * The displayed value lives in node.properties (not in widgets_values), so it
 * can never shift into the `color` widget's slot. The color is validated;
 * an invalid value falls back to Bond purple, and a corrupted saved color
 * (e.g. a number written into it by the old version) is reset on load.
 * Registered as its own extension so the ShowText handler above is untouched.
 * ------------------------------------------------------------------------- */
app.registerExtension({
    name: "BondNodeSuite.BigNumber",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "BondBigNumber") return;

        const BOND_PURPLE  = "#9b30ff";
        const PROP_KEY     = "bond_big_number";
        const WIDGET_NAME  = "__bond_number__";
        const FONT_FAMILY  = "'Arial Black', Arial, sans-serif";
        const MIN_BOX_H    = 60;   // minimum display height (canvas units)
        const DEFAULT_SIZE = [320, 240];

        function isValidColor(v) {
            if (!v) return false;
            if (/^-?\d+(\.\d+)?$/.test(v)) return false;   // bare numbers are never colors
            try { return CSS.supports("color", v); } catch { return false; }
        }

        function colorWidget(node) {
            return node.widgets?.find((w) => w.name === "color");
        }

        function currentColor(node) {
            const v = (colorWidget(node)?.value ?? "").toString().trim();
            return isValidColor(v) ? v : BOND_PURPLE;
        }

        function roundRect(ctx, x, y, w, h, r) {
            ctx.beginPath();
            if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
            else ctx.rect(x, y, w, h);
        }

        // Fit `text` inside the box (x, y, w, h): measure at 100px, scale to the
        // tighter axis, then centre on the glyphs' real ink bounds.
        function drawFittedNumber(ctx, text, x, y, w, h, color) {
            if (!text || w <= 0 || h <= 0) return;
            const pad = Math.max(4, Math.min(w, h) * 0.06);
            const aw  = w - pad * 2;
            const ah  = h - pad * 2;
            if (aw <= 0 || ah <= 0) return;

            ctx.textAlign    = "center";
            ctx.textBaseline = "alphabetic";
            ctx.font = `900 100px ${FONT_FAMILY}`;
            const m  = ctx.measureText(text);
            const tw = Math.max(1, (m.actualBoundingBoxLeft ?? 0) + (m.actualBoundingBoxRight ?? m.width) || m.width);
            const th = Math.max(1, (m.actualBoundingBoxAscent ?? 72) + (m.actualBoundingBoxDescent ?? 0));
            let fs = 100 * Math.min(aw / tw, ah / th);
            if (!isFinite(fs) || fs < 6) fs = 6;

            ctx.font = `900 ${fs}px ${FONT_FAMILY}`;
            const m2   = ctx.measureText(text);
            const asc  = m2.actualBoundingBoxAscent  ?? fs * 0.72;
            const desc = m2.actualBoundingBoxDescent ?? 0;
            const left = m2.actualBoundingBoxLeft    ?? m2.width / 2;
            const rght = m2.actualBoundingBoxRight   ?? m2.width / 2;

            ctx.fillStyle    = color;
            const cx = x + w / 2 + (left - rght) / 2;
            const cy = y + h / 2 + (asc - desc) / 2;
            ctx.fillText(text, cx, cy);
        }

        function setDisplayed(node, text) {
            const value = text === null || text === undefined ? "" : String(text);
            node.properties = node.properties || {};
            node.properties[PROP_KEY] = value;
            const w = node.widgets?.find((x) => x.name === WIDGET_NAME);
            if (w) w.value = value;
            node.setDirtyCanvas?.(true, true);
        }

        // Expose for local testing only; harmless in ComfyUI.
        nodeType.__bondDrawFittedNumber = drawFittedNumber;

        nodeType.prototype.onNodeCreated = (function (orig) {
            return function () {
                const r = orig?.apply(this, arguments);
                const node = this;

                if (!node.widgets?.some((w) => w.name === WIDGET_NAME)) {
                    const widget = {
                        name: WIDGET_NAME,
                        type: "bond_big_number",
                        value: "",
                        serialize: false,                 // never written to widgets_values
                        options: { serialize: false },    // never sent to the backend prompt
                        computeSize(width) {
                            // Fixed minimum; the drawing itself fills all remaining
                            // node height, so there is no width/height feedback loop.
                            return [width, MIN_BOX_H];
                        },
                        draw(ctx, n, widgetWidth, y) {
                            const margin = 10;
                            const x = margin;
                            const w = widgetWidth - margin * 2;
                            const top = y + 2;
                            const h = Math.max(MIN_BOX_H - 4, n.size[1] - top - margin);

                            ctx.save();
                            roundRect(ctx, x, top, w, h, 6);
                            ctx.fillStyle   = "rgba(0,0,0,0.25)";
                            ctx.fill();
                            ctx.strokeStyle = (typeof LiteGraph !== "undefined" && LiteGraph.WIDGET_OUTLINE_COLOR) || "#555";
                            ctx.lineWidth   = 1;
                            ctx.stroke();
                            ctx.clip();
                            drawFittedNumber(ctx, this.value, x, top, w, h, currentColor(n));
                            ctx.restore();
                        },
                    };
                    if (node.addCustomWidget) node.addCustomWidget(widget);
                    else (node.widgets ||= []).push(widget);
                }

                node.properties = node.properties || {};
                if (node.properties[PROP_KEY] === undefined) node.properties[PROP_KEY] = "";

                // Repaint immediately when the color is edited.
                const cw = colorWidget(node);
                if (cw && !cw.__bondHooked) {
                    cw.__bondHooked = true;
                    const origCb = cw.callback;
                    cw.callback = function () {
                        const res = origCb?.apply(this, arguments);
                        node.setDirtyCanvas?.(true, true);
                        return res;
                    };
                }

                if (node.size[0] < DEFAULT_SIZE[0] || node.size[1] < DEFAULT_SIZE[1]) {
                    node.setSize?.([Math.max(node.size[0], DEFAULT_SIZE[0]), Math.max(node.size[1], DEFAULT_SIZE[1])]);
                }
                return r;
            };
        })(nodeType.prototype.onNodeCreated);

        // Render on execution.
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const v = Array.isArray(message?.number)
                ? message.number.flat().find((x) => x !== null && x !== undefined && x !== "")
                : message?.number;
            setDisplayed(this, v);
        };

        // Keep the raw saved widgets_values for legacy migration.
        const VALUES = Symbol();
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function () {
            this[VALUES] = arguments[0]?.widgets_values;
            return configure?.apply(this, arguments);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);

            // Self-heal a color slot that holds a number or other junk.
            const cw = colorWidget(this);
            if (cw) {
                const cv = (cw.value ?? "").toString().trim();
                if (!isValidColor(cv)) cw.value = BOND_PURPLE;
            }

            // Restore the displayed number: properties first (current format),
            // otherwise migrate from an older save that kept it in widgets_values.
            let saved = this.properties?.[PROP_KEY];
            if (!saved) {
                const wv = this[VALUES];
                if (Array.isArray(wv)) {
                    for (let i = wv.length - 1; i >= 0; i--) {
                        const cand = wv[i] === null || wv[i] === undefined ? "" : String(wv[i]).trim();
                        if (/^-?\d+$/.test(cand)) { saved = cand; break; }
                    }
                }
            }
            setDisplayed(this, saved ?? "");
            return r;
        };
    },
});
