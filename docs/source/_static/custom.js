document.addEventListener("DOMContentLoaded", function() {
    window.addEventListener("resize", function() {
        var plotlyDivs = document.getElementsByClassName('plotly-container');
        for (var i = 0; i < plotlyDivs.length; i++) {
            Plotly.Plots.resize(plotlyDivs[i]);
        }
    });
});
