/**
 * Bond Node Suite — Global Bypass Button (One-Way)
 * Adds a button to the ComfyUI toolbar that bypasses every node in the graph
 * except those whose title starts with [Global] or whose type is in the blocklist.
 * This is a one-way push — it only bypasses, it does not restore.
 *
 * Usage: prefix any node title with [Global] to protect it from bypassing.
 * Example: "[Global] KSampler" or "[Global] Bond: Save Image"
 */
import { app } from "../../scripts/app.js";

const KEEP_PREFIX  = "[Global]";
const MODE_BYPASS  = 4; // LiteGraph bypass mode

// Node types that are never bypassed regardless of title or prefix.
// Add any node type here that should always remain active.
const BYPASS_BLOCKLIST = new Set([
    "Note",
    "Label (rgthree)",
    "Fast Groups Bypasser (rgthree)",
    "Fast Groups Muter (rgthree)",
]);

function isProtected(node) {
    const title = (node.title ?? node.type ?? "").trim();
    const type  = (node.type ?? "").trim();
    return title.startsWith(KEEP_PREFIX) || BYPASS_BLOCKLIST.has(type);
}

function applyBypassAll() {
    const nodes = app.graph._nodes ?? [];
    for (const node of nodes) {
        if (isProtected(node)) continue;
        node.mode = MODE_BYPASS;
    }
    app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "BondNodeSuite.GlobalBypassButton",

    async setup() {
        const btn = document.createElement("button");
        btn.id = "bond-bypass-button";

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
        label.textContent = "Bypass All";
        btn.appendChild(label);

        btn.title = "Bypass all nodes except [Global] nodes (Bond Node Suite)";

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
            btn.style.background  = "var(--comfy-input-bg, #1a1a2e)";
            btn.style.borderColor = "var(--border-color, #444)";
            btn.style.color       = "var(--input-text, #ccc)";
        });

        btn.addEventListener("click", () => {
            applyBypassAll();
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
    },
});
