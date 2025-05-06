document.addEventListener("DOMContentLoaded", function () {
    const observer = new MutationObserver(() => {
        // Trigger resize for all Plotly figures
        const plotlyFigures = document.querySelectorAll(".plotly-graph-div");
        plotlyFigures.forEach((figure) => {
            if (window.Plotly) {
                window.Plotly.Plots.resize(figure);
            }
        });
    });

    // Observe changes in the secondary sidebar
    const sidebar = document.querySelector(".bd-sidebar-secondary");
    if (sidebar) {
        observer.observe(sidebar, { attributes: true, childList: true, subtree: true });
    }
});
