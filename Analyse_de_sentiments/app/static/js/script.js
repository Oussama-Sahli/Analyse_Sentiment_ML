function uploadFile() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];
    if (!file) {
        document.getElementById("result").innerHTML = "<p class='error'>Veuillez sélectionner un fichier.</p>";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    $.ajax({
        url: "/predict",
        type: "POST",
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            // Lorsque le fichier est traité, afficher le lien pour télécharger le fichier résultant
            const downloadLink = `<a href="/download" download="predictions.csv">Télécharger les résultats</a>`;
            document.getElementById("result").innerHTML = downloadLink;
        },
        error: function(xhr, status, error) {
            const errorMsg = xhr.responseJSON && xhr.responseJSON.error ? xhr.responseJSON.error : "Une erreur est survenue.";
            document.getElementById("result").innerHTML = `<p class='error'>${errorMsg}</p>`;
        }
    });
}
