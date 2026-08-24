(function () {
  "use strict";

  const MIN_HEIGHT = 320;
  const MAX_HEIGHT = 1600;
  let requestedWidth = 100;
  let requestedHeightScale = 1;
  let observer = null;
  let scheduled = false;

  function clamp(value, minimum, maximum, fallback) {
    const numeric = Number(value);
    return Number.isFinite(numeric)
      ? Math.min(maximum, Math.max(minimum, numeric))
      : fallback;
  }

  function graphRoots() {
    const page = document.getElementById("barracuda-page");
    if (!page) return [];
    const roots = [];
    page.querySelectorAll(".js-plotly-plot").forEach((plot) => {
      if (plot.parentElement && !roots.includes(plot.parentElement)) {
        roots.push(plot.parentElement);
      }
    });
    return roots;
  }

  function naturalHeight(root, plot) {
    const inlineHeight = root.style.height || "";
    // Dash uses `height: 100%` on graphs that should inherit their Plotly
    // layout height.  Treat only explicit pixel heights as a server-provided
    // natural size; parsing the percentage as 100 would collapse the figure.
    const current = /px\s*$/i.test(inlineHeight)
      ? parseFloat(inlineHeight)
      : Number.NaN;
    const applied = parseFloat(root.dataset.barracudaAppliedHeight || "");
    let stored = parseFloat(root.dataset.barracudaBaseHeight || "");

    // A Dash callback can replace an existing graph's inline height. Treat a
    // value that differs from our last applied height as its new natural size.
    if (Number.isFinite(current) && (!Number.isFinite(applied) || Math.abs(current - applied) > 1)) {
      stored = current;
      root.dataset.barracudaBaseHeight = String(stored);
    }
    if (!Number.isFinite(stored)) {
      const plotHeight = Number(plot && plot._fullLayout && plot._fullLayout.height);
      const renderedHeight = root.getBoundingClientRect().height;
      stored = Number.isFinite(plotHeight) && plotHeight > 0
        ? plotHeight
        : renderedHeight > 0
          ? renderedHeight
          : 430;
      root.dataset.barracudaBaseHeight = String(stored);
    }
    return stored;
  }

  function resizeGraph(root) {
    const plot = root.querySelector(".js-plotly-plot");
    if (!plot) return;
    const rootRect = root.getBoundingClientRect();
    if (root.offsetParent === null || rootRect.width <= 0) return;
    const mobile = window.matchMedia("(max-width: 820px)").matches;
    const width = mobile ? 100 : requestedWidth;
    const baseHeight = naturalHeight(root, plot);
    const height = Math.round(clamp(baseHeight * requestedHeightScale, MIN_HEIGHT, MAX_HEIGHT, 430));
    const widthValue = `${width}%`;
    const heightValue = `${height}px`;

    if (root.style.width !== widthValue) root.style.width = widthValue;
    if (root.style.maxWidth !== "100%") root.style.maxWidth = "100%";
    if (root.style.marginLeft !== "auto") root.style.marginLeft = "auto";
    if (root.style.marginRight !== "auto") root.style.marginRight = "auto";
    if (root.style.height !== heightValue) root.style.height = heightValue;
    if (plot.style.width !== "100%") plot.style.width = "100%";
    if (plot.style.height !== heightValue) plot.style.height = heightValue;
    root.dataset.barracudaAppliedHeight = String(height);
    root.classList.add("barracuda-figure-sized");

    window.requestAnimationFrame(function () {
      if (window.Plotly && window.Plotly.Plots && plot.isConnected && plot._fullLayout) {
        try {
          const resized = window.Plotly.Plots.resize(plot);
          if (resized && typeof resized.catch === "function") resized.catch(function () {});
        } catch (error) {
          // A graph may be removed between scheduling and the animation frame.
          // The MutationObserver will size its replacement when it mounts.
        }
      }
    });
  }

  function applyAll() {
    scheduled = false;
    graphRoots().forEach(resizeGraph);
  }

  function schedule() {
    if (scheduled) return;
    scheduled = true;
    window.requestAnimationFrame(applyAll);
  }

  function ensureObserver() {
    const page = document.getElementById("barracuda-page");
    if (!page) return;
    if (observer && observer._barracudaPage === page) return;
    if (observer) observer.disconnect();
    observer = new MutationObserver(function (records) {
      const relevant = records.some(function (record) {
        return record.type === "childList" ||
          (record.type === "attributes" && record.target.parentElement === page) ||
          (record.type === "attributes" && record.target.querySelector && record.target.querySelector(".js-plotly-plot"));
      });
      if (relevant) schedule();
    });
    observer._barracudaPage = page;
    observer.observe(page, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["style", "class"],
    });
  }

  window.addEventListener("resize", schedule, { passive: true });

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    barracudaFigureControls: {
      apply: function (width, heightScale, pathname) {
        requestedWidth = clamp(width, 60, 100, 100);
        requestedHeightScale = clamp(heightScale, 0.75, 1.75, 1);
        ensureObserver();
        schedule();
        window.setTimeout(schedule, 120);
        window.setTimeout(schedule, 500);
        return {
          width: requestedWidth,
          heightScale: requestedHeightScale,
          pathname: pathname || "/",
          appliedAt: Date.now(),
        };
      },
    },
  });
})();
