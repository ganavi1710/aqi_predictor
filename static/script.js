// ========================================================
// SMART AIR QUALITY MONITORING & PREDICTION SYSTEM
// Multi-Model Client Controller (Leaflet Map, Realtime Telemetry,
// Multi-Algorithm Prediction APIs, Comparison Charts, History)
// ========================================================

// 1. GLOBAL STATE & TELEMETRY
let liveTelemetry = {
    pm25: null,
    co: null,
    no2: null,
    temperature: null,
    humidity: null,
    wind: null,
    city: "Detecting..."
};

// 2. LEAFLET MAP INITIALIZATION
const map = L.map('map').setView([20, 78], 5);

L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19
}).addTo(map);

let marker = L.marker([20, 78]).addTo(map);

setTimeout(() => { map.invalidateSize(); }, 300);
window.addEventListener('resize', () => { map.invalidateSize(); });

// 3. REAL-TIME TREND CHART (Chart.js)
let aqiData = [];
let labels = [];
const chartCanvas = document.getElementById('chart');
let chart = null;

if (chartCanvas) {
    chart = new Chart(chartCanvas, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'AQI Level',
                data: [],
                borderColor: '#00ff88',
                backgroundColor: 'rgba(0, 255, 136, 0.15)',
                fill: true,
                tension: 0.35,
                borderWidth: 3,
                pointRadius: 4,
                pointBackgroundColor: '#00ff88'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#dce9f5', font: { family: 'Poppins', size: 12 } }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#8da9c4', font: { family: 'Poppins' } },
                    grid: { color: 'rgba(255, 255, 255, 0.06)' }
                },
                y: {
                    ticks: { color: '#8da9c4', font: { family: 'Poppins' } },
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    beginAtZero: true
                }
            }
        }
    });
}

// 4. REALTIME TELEMETRY FETCH & UPDATE
let activeLat = 20;
let activeLon = 78;

function updateAQI(lat, lon) {
    activeLat = lat;
    activeLon = lon;
    const statusBox = document.getElementById("statusBox");
    if (statusBox) statusBox.innerText = "Fetching live telemetry...";

    fetch(`/realtime?lat=${lat}&lon=${lon}&t=${Date.now()}`)
        .then(res => res.json())
        .then(data => {
            if (!data || data.aqi === "Error") {
                document.getElementById("liveAQI").innerText = "--";
                if (statusBox) statusBox.innerText = "Telemetry unavailable";
                document.getElementById("cityName").innerHTML = `<i class="fa-solid fa-location-crosshairs"></i> ${data.city || "Location error"}`;
                return;
            }

            liveTelemetry = {
                pm25: data.pm25,
                co: data.co,
                no2: data.no2,
                temperature: data.temp,
                humidity: data.humidity,
                wind: data.wind,
                city: data.city
            };

            const aqi = Number(data.aqi);
            const color = data.color || '#00ff88';

            document.getElementById("liveAQI").innerText = Math.round(aqi);
            document.getElementById("liveAQI").style.color = color;

            if (statusBox) {
                statusBox.innerText = data.status || data.category || 'Good';
                statusBox.style.color = color;
            }

            document.getElementById("cityName").innerHTML = `<i class="fa-solid fa-location-dot"></i> ${data.city}`;
            document.getElementById("temp").innerText = `${Number(data.temp).toFixed(1)}°C`;
            document.getElementById("extra").innerText = `Humidity: ${data.humidity}% | Wind: ${data.wind} km/h`;

            const pmElem = document.getElementById("livePM25");
            if (pmElem) pmElem.innerText = data.pm25;
            const coElem = document.getElementById("liveCO");
            if (coElem) coElem.innerText = data.co;
            const no2Elem = document.getElementById("liveNO2");
            if (no2Elem) no2Elem.innerText = data.no2;

            // Machine Learning Projection Card
            const predAQIElem = document.getElementById("predictedAQI");
            if (predAQIElem && data.predicted_aqi !== undefined) {
                predAQIElem.innerText = data.predicted_aqi;
                const catElem = document.getElementById("predictionCategory");
                if (catElem) {
                    catElem.innerText = data.predicted_badge || data.predicted_category || "Moderate";
                }
                const tipElem = document.getElementById("predictionTip");
                if (tipElem) {
                    tipElem.innerText = data.predicted_recommendation || data.recommendation || "Maintain healthy airflow.";
                }
            }

            const circle = document.querySelector('.aqi-circle');
            if (circle) {
                circle.style.boxShadow = `0 0 35px ${color}`;
            }

            if (chart) {
                if (data.trend_labels && data.trend_labels.length > 0) {
                    labels = data.trend_labels;
                    aqiData = data.trend_values;
                } else {
                    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                    aqiData.push(aqi);
                    labels.push(time);
                    if (aqiData.length > 12) {
                        aqiData.shift();
                        labels.shift();
                    }
                }

                chart.data.labels = labels;
                chart.data.datasets[0].data = aqiData;
                chart.data.datasets[0].borderColor = color;
                chart.data.datasets[0].pointBackgroundColor = color;
                chart.update();
            }
        })
        .catch(err => {
            console.error("AQI telemetry fetch failed:", err);
            if (statusBox) statusBox.innerText = "Network timeout";
        });
}

// 4.1 PERIODIC LIVE TELEMETRY REFRESH (Every 30 seconds)
setInterval(() => {
    if (activeLat !== null && activeLon !== null) {
        updateAQI(activeLat, activeLon);
    }
}, 30000);

// 5. MAP CLICK & GEOLOCATION
map.on('click', function(e) {
    const lat = e.latlng.lat;
    const lon = e.latlng.lng;
    marker.setLatLng([lat, lon]);
    updateAQI(lat, lon);
});

if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
        function(pos) {
            const lat = pos.coords.latitude;
            const lon = pos.coords.longitude;
            map.setView([lat, lon], 10);
            marker.setLatLng([lat, lon]);
            updateAQI(lat, lon);
        },
        function(err) {
            console.warn("Geolocation denied or unavailable:", err.message);
            updateAQI(20, 78);
        }
    );
} else {
    updateAQI(20, 78);
}

// 6. AUTOFILL "USE CURRENT AIR QUALITY"
const btnUseCurrent = document.getElementById("btnUseCurrent");
if (btnUseCurrent) {
    btnUseCurrent.addEventListener("click", function() {
        if (liveTelemetry.pm25 === null) {
            alert("Live telemetry is loading. Click a point on the map or wait a few seconds.");
            return;
        }

        document.getElementById("input_pm25").value = liveTelemetry.pm25;
        document.getElementById("input_co").value = liveTelemetry.co;
        document.getElementById("input_no2").value = liveTelemetry.no2;
        document.getElementById("input_temp").value = liveTelemetry.temperature;
        document.getElementById("input_hum").value = liveTelemetry.humidity;
        document.getElementById("input_wind").value = liveTelemetry.wind;

        const statusMsg = document.getElementById("predictStatusMsg");
        if (statusMsg) {
            statusMsg.innerHTML = `<span style="color:#00ff88;"><i class="fa-solid fa-check"></i> Populated from ${liveTelemetry.city}</span>`;
            setTimeout(() => { statusMsg.innerHTML = ""; }, 4000);
        }
    });
}

// 7. MULTI-MODEL PREDICTION FORM SUBMISSION
const predictionForm = document.getElementById("predictionForm");
if (predictionForm) {
    predictionForm.addEventListener("submit", function(e) {
        e.preventDefault();

        const btnSubmit = document.getElementById("btnSubmitPredict");
        const statusMsg = document.getElementById("predictStatusMsg");
        const originalBtnText = btnSubmit.innerHTML;

        btnSubmit.disabled = true;
        btnSubmit.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Evaluating Model...`;
        if (statusMsg) statusMsg.innerText = "";

        const selectedModel = document.getElementById("model_select").value;

        const payload = {
            model: selectedModel,
            pm25: parseFloat(document.getElementById("input_pm25").value),
            co: parseFloat(document.getElementById("input_co").value),
            no2: parseFloat(document.getElementById("input_no2").value),
            temperature: document.getElementById("input_temp").value ? parseFloat(document.getElementById("input_temp").value) : null,
            humidity: document.getElementById("input_hum").value ? parseFloat(document.getElementById("input_hum").value) : null,
            wind_speed: document.getElementById("input_wind").value ? parseFloat(document.getElementById("input_wind").value) : null,
            location: liveTelemetry.city || "Custom Input"
        };

        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            btnSubmit.disabled = false;
            btnSubmit.innerHTML = originalBtnText;

            if (!data.success) {
                if (statusMsg) {
                    statusMsg.innerHTML = `<span style="color:#ff4d6d;"><i class="fa-solid fa-circle-exclamation"></i> ${data.error || 'Prediction failed'}</span>`;
                }
                return;
            }

            const resultBox = document.getElementById("predictionResultBox");
            if (resultBox) {
                resultBox.style.display = "block";
                resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

                document.getElementById("resAQI").innerText = data.predicted_aqi;
                document.getElementById("resAQI").style.color = data.color || '#00ff88';

                const badge = document.getElementById("resCategoryBadge");
                badge.innerText = data.badge;
                badge.style.background = `${data.color}25`;
                badge.style.color = data.color;
                badge.style.border = `1px solid ${data.color}50`;

                document.getElementById("resRecommendation").innerText = data.recommendation;

                const modelBadge = document.getElementById("resModelBadge");
                modelBadge.innerHTML = `<i class="fa-solid fa-microchip"></i> Model: <strong>${data.model}</strong>`;

                const bestTag = document.getElementById("resBestTag");
                if (bestTag) {
                    bestTag.style.display = data.is_best_model ? "inline-flex" : "none";
                }

                const tipsList = document.getElementById("resTipsList");
                tipsList.innerHTML = "";
                (data.tips || []).forEach(tip => {
                    const li = document.createElement("li");
                    li.innerHTML = `<i class="fa-solid fa-check" style="color:${data.color};"></i> ${tip}`;
                    tipsList.appendChild(li);
                });

                document.getElementById("resTimestamp").innerText = new Date().toLocaleTimeString();
            }

            loadPredictionHistory();
        })
        .catch(err => {
            btnSubmit.disabled = false;
            btnSubmit.innerHTML = originalBtnText;
            if (statusMsg) {
                statusMsg.innerHTML = `<span style="color:#ff4d6d;"><i class="fa-solid fa-triangle-exclamation"></i> Network communication error</span>`;
            }
            console.error("Prediction submission error:", err);
        });
    });
}

// 8. INTERACTIVE MODEL DIAGNOSTICS VIEWER
function updateDiagnosticPlots(modelKey) {
    const imgActualPred = document.getElementById("imgActualPred");
    const imgResiduals = document.getElementById("imgResiduals");
    const titleActual = document.getElementById("diagActualPredTitle");
    const titleResidual = document.getElementById("diagResidualTitle");

    const prettyName = document.querySelector(`#diagnosticModelSelect option[value="${modelKey}"]`).innerText;

    if (imgActualPred) {
        imgActualPred.src = `/static/models/${modelKey}_actual_pred.png?t=${Date.now()}`;
    }
    if (imgResiduals) {
        imgResiduals.src = `/static/models/${modelKey}_residuals.png?t=${Date.now()}`;
    }
    if (titleActual) {
        titleActual.innerText = `${prettyName} — Actual vs Predicted`;
    }
    if (titleResidual) {
        titleResidual.innerText = `${prettyName} — Residual Distribution`;
    }
}

// 9. MODEL COMPARISON CHARTS INITIALIZATION (Chart.js)
function initComparisonCharts() {
    fetch('/api/model-comparison')
        .then(res => res.json())
        .then(data => {
            if (!data.success || !data.metadata || !data.metadata.models) return;

            const modelsObj = data.metadata.models;
            const bestKey = data.metadata.best_model;

            const labels = [];
            const rmseData = [];
            const maeData = [];
            const r2Data = [];
            const bgColors = [];

            // Sort models by RMSE ascending
            const sortedKeys = Object.keys(modelsObj).sort((a, b) => {
                return modelsObj[a].test_metrics.RMSE - modelsObj[b].test_metrics.RMSE;
            });

            sortedKeys.forEach(key => {
                const m = modelsObj[key];
                labels.push(m.name.replace(' Regressor', ''));
                rmseData.push(m.test_metrics.RMSE);
                maeData.push(m.test_metrics.MAE);
                r2Data.push(m.test_metrics.R2_Score);

                if (key === bestKey) {
                    bgColors.push('#00ff88');  // Winner highlight
                } else {
                    bgColors.push('#00bfff');
                }
            });

            const commonOptions = {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        ticks: { color: '#8da9c4', font: { family: 'Poppins', size: 10 } },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' }
                    },
                    y: {
                        ticks: { color: '#8da9c4', font: { family: 'Poppins', size: 10 } },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' }
                    }
                }
            };

            // 1. RMSE Chart
            const ctxRMSE = document.getElementById('chartRMSE');
            if (ctxRMSE) {
                new Chart(ctxRMSE, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            data: rmseData,
                            backgroundColor: bgColors,
                            borderRadius: 6
                        }]
                    },
                    options: commonOptions
                });
            }

            // 2. MAE Chart
            const ctxMAE = document.getElementById('chartMAE');
            if (ctxMAE) {
                new Chart(ctxMAE, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            data: maeData,
                            backgroundColor: bgColors,
                            borderRadius: 6
                        }]
                    },
                    options: commonOptions
                });
            }

            // 3. R2 Chart
            const ctxR2 = document.getElementById('chartR2');
            if (ctxR2) {
                new Chart(ctxR2, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            data: r2Data,
                            backgroundColor: bgColors,
                            borderRadius: 6
                        }]
                    },
                    options: {
                        ...commonOptions,
                        scales: {
                            ...commonOptions.scales,
                            y: {
                                min: 0.8,
                                max: 1.0,
                                ticks: { color: '#8da9c4' },
                                grid: { color: 'rgba(255, 255, 255, 0.05)' }
                            }
                        }
                    }
                });
            }
        })
        .catch(err => console.error("Failed to load model comparison charts:", err));
}

// 10. PREDICTION HISTORY LOADER
function loadPredictionHistory() {
    fetch('/api/history?limit=15')
        .then(res => res.json())
        .then(data => {
            if (!data.success || !data.history) return;

            const tbody = document.getElementById("historyTableBody");
            if (!tbody) return;

            if (data.history.length === 0) {
                tbody.innerHTML = `<tr><td colspan="10" style="text-align:center; padding: 25px; color: #9db2c7;">No prediction records yet.</td></tr>`;
                return;
            }

            tbody.innerHTML = "";
            data.history.forEach(rec => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td class="mono">${rec.timestamp}</td>
                    <td>${rec.location}</td>
                    <td><span class="badge-model-cell">${rec.model_used}</span></td>
                    <td>${rec.pm25}</td>
                    <td>${rec.co}</td>
                    <td>${rec.no2}</td>
                    <td>${rec.temperature !== null ? rec.temperature + '°C' : '--'} | ${rec.humidity !== null ? rec.humidity + '%' : '--'}</td>
                    <td><strong>${rec.predicted_aqi}</strong></td>
                    <td><span class="badge-cat">${rec.category}</span></td>
                    <td class="rec-cell" title="${rec.recommendation}">${rec.recommendation}</td>
                `;
                tbody.appendChild(tr);
            });
        })
        .catch(err => console.error("Error refreshing history:", err));
}

// 11. LIVE CLOCK
function updateClock() {
    const clockElem = document.getElementById("clock");
    if (clockElem) {
        clockElem.innerText = new Date().toLocaleTimeString();
    }
}
setInterval(updateClock, 1000);
updateClock();

// 12. SMOOTH SCROLLING
function scrollToSection(id, element) {
    const target = document.getElementById(id);
    if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    document.querySelectorAll('.sidebar ul li').forEach(li => {
        li.classList.remove('active');
    });

    if (element) {
        element.classList.add('active');
    }
}

// Initialize comparison charts on load
document.addEventListener('DOMContentLoaded', function() {
    initComparisonCharts();
});