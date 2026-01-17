async function predict() {
    const data = {
        duration_hours: parseFloat(document.getElementById("duration").value),
        hour_of_day: parseInt(document.getElementById("hour").value),
        day_of_year: parseInt(document.getElementById("day").value),
        kp_index_lag1: parseFloat(document.getElementById("lag").value),
        event_type: document.getElementById("event").value,
        class_type: document.getElementById("class").value
    };

    const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    if (result.predicted_kp_index !== undefined) {
        document.getElementById("result").innerText =
            "Predicted KP Index: " + result.predicted_kp_index;
    } else {
        document.getElementById("result").innerText =
            "Error: " + result.error;
    }
}
