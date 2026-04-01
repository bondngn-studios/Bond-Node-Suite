/**
 * Bond Node Suite — Global Bypass Toggle
 * Adds a button to the ComfyUI toolbar that bypasses every node in the graph
 * except those whose title starts with [Global]. Click again to restore all.
 * State persists across page reloads via localStorage.
 *
 * Usage: prefix any node title with [Global] to protect it from bypassing.
 * Example: "[Global] KSampler" or "[Global] Bond: Save Image"
 */
import { app } from "../../scripts/app.js";

const STORAGE_KEY  = "bond_bypass_active";
const KEEP_PREFIX  = "[Global]";
const MODE_BYPASS  = 4; // LiteGraph bypass mode
const MODE_NORMAL  = 0; // LiteGraph normal mode

// Node types that are never bypassed regardless of title or prefix.
// Add any node type here that should always remain active.
const BYPASS_BLOCKLIST = new Set([
    "Note",
    "Label (rgthree)",
    "Fast Groups Bypasser (rgthree)",
    "Fast Groups Muter (rgthree)",
]);

// Per-node previous mode — stored so we only restore what we actually changed
const _prevModes = new Map();

function isProtected(node) {
    const title = (node.title ?? node.type ?? "").trim();
    const type  = (node.type ?? "").trim();
    return title.startsWith(KEEP_PREFIX) || BYPASS_BLOCKLIST.has(type);
}

function applyBypassAll() {
    const nodes = app.graph._nodes ?? [];
    _prevModes.clear();
    for (const node of nodes) {
        if (isProtected(node)) continue;
        _prevModes.set(node.id, node.mode ?? MODE_NORMAL);
        node.mode = MODE_BYPASS;
    }
    app.graph.setDirtyCanvas(true, true);
}

function applyRestoreAll() {
    const nodes = app.graph._nodes ?? [];
    for (const node of nodes) {
        if (isProtected(node)) continue;
        const prev = _prevModes.get(node.id);
        node.mode = (prev !== undefined) ? prev : MODE_NORMAL;
    }
    _prevModes.clear();
    app.graph.setDirtyCanvas(true, true);
}

function setButtonState(btn, active) {
    const label = btn.querySelector("span");
    if (active) {
        if (label) label.textContent = "Restore";
        btn.title          = "Restore all bypassed nodes (Bond Node Suite)";
        btn.dataset.active = "true";
    } else {
        if (label) label.textContent = "Bypass";
        btn.title          = "Bypass all nodes except [Global] nodes (Bond Node Suite)";
        btn.dataset.active = "false";
    }
}

app.registerExtension({
    name: "BondNodeSuite.GlobalBypassToggle",

    async setup() {
        const btn = document.createElement("button");
        btn.id = "bond-bypass-toggle";

        // Match ComfyUI native button style
        btn.style.cssText = [
            "display:inline-flex",
            "align-items:center",
            "gap:4px",
            "padding:0 10px",
            "height:32px",
            "border-radius:6px",
            "border:1px solid var(--border-color, #444)",
            "background:var(--comfy-input-bg, #1a1a2e)",
            "color:var(--input-text, #ccc)",
            "cursor:pointer",
            "font-size:12px",
            "font-family:inherit",
            "font-weight:500",
            "white-space:nowrap",
            "transition:background 0.15s, color 0.15s, border-color 0.15s",
            "margin-left:4px",
        ].join(";");

        btn.addEventListener("mouseenter", () => {
            btn.style.background  = "var(--comfy-menu-bg, #2a2a3e)";
            btn.style.borderColor = "var(--primary-color, #9b30ff)";
        });
        btn.addEventListener("mouseleave", () => {
            if (btn.dataset.active === "true") {
                btn.style.background  = "var(--primary-color, #9b30ff)";
                btn.style.borderColor = "var(--primary-color, #9b30ff)";
                btn.style.color       = "#fff";
            } else {
                btn.style.background  = "var(--comfy-input-bg, #1a1a2e)";
                btn.style.borderColor = "var(--border-color, #444)";
                btn.style.color       = "var(--input-text, #ccc)";
            }
        });

        // Bond icon — purple diamond with B
        const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        icon.setAttribute("width", "14");
        icon.setAttribute("height", "14");
        icon.setAttribute("viewBox", "0 0 14 14");
        icon.setAttribute("fill", "none");
        icon.style.flexShrink = "0";
        icon.innerHTML = `
            <polygon points="7,1 13,7 7,13 1,7" fill="#9b30ff" stroke="#c97bff" stroke-width="0.5"/>
            <text x="7" y="10" text-anchor="middle" font-family="Arial Black, sans-serif"
                  font-size="7" font-weight="900" fill="white" letter-spacing="-0.5">B</text>
        `;
        btn.appendChild(icon);

        const label = document.createElement("span");
        btn.appendChild(label);

        let active = localStorage.getItem(STORAGE_KEY) === "true";
        setButtonState(btn, active);

        if (active) {
            btn.style.background  = "var(--primary-color, #9b30ff)";
            btn.style.borderColor = "var(--primary-color, #9b30ff)";
            btn.style.color       = "#fff";
        }

        btn.addEventListener("click", () => {
            active = !active;
            localStorage.setItem(STORAGE_KEY, String(active));
            setButtonState(btn, active);

            if (active) {
                btn.style.background  = "var(--primary-color, #9b30ff)";
                btn.style.borderColor = "var(--primary-color, #9b30ff)";
                btn.style.color       = "#fff";
                applyBypassAll();
            } else {
                btn.style.background  = "var(--comfy-input-bg, #1a1a2e)";
                btn.style.borderColor = "var(--border-color, #444)";
                btn.style.color       = "var(--input-text, #ccc)";
                applyRestoreAll();
            }
        });

        // Inject as leftmost item in the top bar button container
        const inject = () => {
            const container = document.querySelector(".flex.gap-2.mx-2");
            if (container) {
                container.insertBefore(btn, container.firstElementChild);
                return true;
            }
            return false;
        };

        if (!inject()) {
            const observer = new MutationObserver(() => {
                if (inject()) observer.disconnect();
            });
            observer.observe(document.body, { childList: true, subtree: true });
        }

        // Re-apply bypass state after a workflow loads
        let applied = false;
        const tryReapply = () => {
            if (applied) return;
            if (active && (app.graph._nodes?.length ?? 0) > 0) {
                applied = true;
                applyBypassAll();
            }
        };

        const origConfigure = app.graph.configure?.bind(app.graph);
        if (origConfigure) {
            app.graph.configure = function(...args) {
                const result = origConfigure(...args);
                applied = false;
                requestAnimationFrame(tryReapply);
                return result;
            };
        }

        setTimeout(tryReapply, 1000);
    },
});
