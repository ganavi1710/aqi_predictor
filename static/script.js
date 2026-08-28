// =========================
// MAP
// =========================

const map = L.map('map').setView([20, 78], 5);

L.tileLayer(
    'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
).addTo(map);


// =========================
// MARKER
// =========================

let marker = L.marker([20, 78]).addTo(map);


// =========================
// AQI HISTORY
// =========================

let aqiData = [];
let labels = [];


// =========================
// CHART
// =========================

const chart = new Chart(
    document.getElementById('chart'),
    {
        type: 'line',

        data: {
            labels: [],

            datasets: [{
                label: 'AQI',

                data: [],

                borderColor: '#00ff88',

                backgroundColor: 'rgba(0,255,136,0.2)',

                fill: true,

                tension: 0.4,

                borderWidth: 3,

                pointRadius: 5
            }]
        },

        options: {

            responsive: true,

            plugins: {

                legend: {

                    labels: {
                        color: 'white'
                    }
                }
            },

            scales: {

                x: {

                    ticks: {
                        color: 'white'
                    }
                },

                y: {

                    ticks: {
                        color: 'white'
                    }
                }
            }
        }
    }
);


// =========================
// UPDATE AQI
// =========================

function updateAQI(lat, lon) {

    console.log("Fetching AQI:", lat, lon);

    fetch(`/realtime?lat=${lat}&lon=${lon}&t=${Date.now()}`)

        .then(res => res.json())

        .then(data => {

            console.log("AQI DATA:", data);


            // =========================
            // ERROR CHECK
            // =========================

            if (
                !data.aqi ||
                data.aqi === "Error"
            ) {

                document.getElementById("liveAQI").innerText = "--";

                document.getElementById("statusBox").innerText =
                    "AQI unavailable";

                document.getElementById("cityName").innerText =
                    "📍 " + (data.city || "Location");

                return;
            }


            // =========================
            // AQI
            // =========================

            const aqi = Number(data.aqi);


            document.getElementById("liveAQI").innerText =
                Math.round(aqi);


            document.getElementById("cityName").innerText =
                "📍 " + data.city;


            // =========================
            // STATUS
            // =========================

            let status = "Good 😊";

            let color = "#00ff88";

            let insight =
                "Air quality is safe in your area.";

            let tips = `
                <li>✔ Ideal for outdoor activities</li>
                <li>✔ Keep windows open</li>
                <li>✔ Stay hydrated</li>
            `;


            if (aqi <= 50) {

                status = "Good 😊";

                color = "#00ff88";

                insight =
                    "Air quality is safe and suitable for outdoor activities.";

                tips = `
                    <li>✔ Ideal for outdoor activities</li>
                    <li>✔ Keep windows open</li>
                    <li>✔ Stay hydrated</li>
                `;
            }


            else if (aqi <= 100) {

                status = "Moderate 😐";

                color = "#ffee00";

                insight =
                    "Air quality is acceptable, but sensitive people should limit prolonged exposure.";

                tips = `
                    <li>✔ Reduce prolonged outdoor exercise</li>
                    <li>✔ Stay hydrated</li>
                    <li>✔ Monitor air quality</li>
                `;
            }


            else if (aqi <= 150) {

                status = "Poor 😷";

                color = "#ff8800";

                insight =
                    "Air pollution is increasing in this region.";

                tips = `
                    <li>✔ Wear a mask outdoors</li>
                    <li>✔ Avoid long outdoor travel</li>
                    <li>✔ Keep indoor air clean</li>
                `;
            }


            else {

                status = "Unhealthy 🚨";

                color = "#ff0040";

                insight =
                    "Air quality is unhealthy. Avoid prolonged outdoor exposure.";

                tips = `
                    <li>✔ Stay indoors when possible</li>
                    <li>✔ Use an air purifier</li>
                    <li>✔ Avoid intense physical activity</li>
                `;
            }


            // =========================
            // UPDATE STATUS
            // =========================

            document.getElementById("statusBox").innerText =
                status;

            document.getElementById("statusBox").style.color =
                color;

            document.getElementById("liveAQI").style.color =
                color;


            // =========================
            // WEATHER
            // =========================

            document.getElementById("temp").innerText =
                Number(data.temp).toFixed(1) + "°C";


            document.getElementById("extra").innerText =
                `Humidity: ${data.humidity}% | Wind: ${data.wind} km/h`;


            // =========================
            // FUTURE AQI
            // =========================

            const predicted =
                Number(data.predicted_aqi);


            document.getElementById("predictedAQI").innerText =
                Math.round(predicted);


            let predictText =
                "Expected AQI will remain stable.";


            if (predicted > 100) {

                predictText =
                    "Air quality may worsen in upcoming hours.";
            }


            document.getElementById("predictionStatus").innerText =
                predictText;


            // =========================
            // AI INSIGHT
            // =========================

            document.getElementById("aiInsight").innerText =
                insight;


            // =========================
            // RECOMMENDATIONS
            // =========================

            const tipsElement =
                document.querySelector(".tips");

            if (tipsElement) {

                tipsElement.innerHTML =
                    tips;
            }


            // =========================
            // AQI CHART
            // =========================

            const time =
                new Date().toLocaleTimeString();


            aqiData.push(aqi);

            labels.push(time);


            if (aqiData.length > 7) {

                aqiData.shift();

                labels.shift();
            }


            chart.data.labels =
                labels;

            chart.data.datasets[0].data =
                aqiData;


            chart.update();


            // =========================
            // GLOW
            // =========================

            const circle =
                document.querySelector('.aqi-circle');


            if (circle) {

                circle.style.boxShadow =
                    `0 0 40px ${color}`;
            }

        })

        .catch(error => {

            console.error(
                "AQI FETCH ERROR:",
                error
            );

        });
}


// =========================
// MAP CLICK
// =========================

map.on('click', function(e) {

    const lat =
        e.latlng.lat;

    const lon =
        e.latlng.lng;


    marker.setLatLng([
        lat,
        lon
    ]);


    updateAQI(
        lat,
        lon
    );

});


// =========================
// USER LOCATION
// =========================

if (navigator.geolocation) {

    navigator.geolocation.getCurrentPosition(

        function(pos) {

            const lat =
                pos.coords.latitude;

            const lon =
                pos.coords.longitude;


            console.log(
                "USER LOCATION:",
                lat,
                lon
            );


            map.setView(
                [lat, lon],
                10
            );


            marker.setLatLng([
                lat,
                lon
            ]);


            updateAQI(
                lat,
                lon
            );

        },

        function(error) {

            console.log(
                "Location permission failed:",
                error
            );


            updateAQI(
                20,
                78
            );
        }
    );

}

else {

    updateAQI(
        20,
        78
    );
}


// =========================
// LIVE CLOCK
// =========================

function updateClock() {

    const now =
        new Date();


    const clock =
        document.getElementById("clock");


    if (clock) {

        clock.innerText =
            now.toLocaleString();
    }
}


setInterval(
    updateClock,
    1000
);


updateClock();


// =========================
// SIDEBAR NAVIGATION
// =========================

function scrollToSection(
    id,
    element
) {

    document
        .getElementById(id)
        .scrollIntoView({
            behavior: 'smooth'
        });


    document
        .querySelectorAll('.sidebar ul li')
        .forEach(li => {

            li.classList.remove(
                'active'
            );

        });


    element.classList.add(
        'active'
    );
}