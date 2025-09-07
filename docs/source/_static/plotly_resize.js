document.addEventListener("DOMContentLoaded", function () {
    const sidebar = document.querySelector(".bd-sidebar-secondary");

    function resizePlotlyFigures() {
        const plotlyFigures = document.querySelectorAll(".plotly-graph-div");
        plotlyFigures.forEach((figure) => {
            if (window.Plotly) {
                window.Plotly.Plots.resize(figure);
            }
        });
    }

    if (sidebar) {
        // Wait for the sidebar to fully load
        const observer = new MutationObserver(() => {
            resizePlotlyFigures();
        });

        observer.observe(sidebar, { attributes: true, childList: true, subtree: true });

        // Trigger an initial resize after a short delay to ensure everything is loaded
        setTimeout(resizePlotlyFigures, 500);
    } else {
        // Fallback: Resize figures if the sidebar is not found
        resizePlotlyFigures();
    }
});
