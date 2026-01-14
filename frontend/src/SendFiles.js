import { showModel } from "./Show3DModel";

export function handleFolderSelect(event) {

    let selectedFiles = Array.from(event.target.files);
    console.log(`Выбрано файлов: ${selectedFiles.length}`);

    if (selectedFiles.length === 0) {
        alert('Выберите папку с файлами');
        return
    }

    /* const notDICOMFiles = selectedFiles.filter(file => !isDcmFile(file));
    
    if (notDICOMFiles.length > 0) {
        alert('Все файлы в папке должны быть в формате .dcm!');
        selectedFiles = [];
        return
    } */

    sendFiles(selectedFiles)
}


function sendFiles(files) {
    console.log('Отправка...');

    const formData = new FormData();
    files.forEach(file => {
        formData.append('files[]', file);
    });

    /* fetch('url', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('Данные с сервера получены');
        showModel(data)
    })
    .catch(error => {
        alert('Ошибка отправки файлов! Попробуйте еще раз');
        console.log('Ошибка', error);
    }); */
    showModel(files)

    return
}


function isDcmFile(file) {
    const extension = file.name.toLowerCase().split('.').pop();
    
    const mimeType = file.type;
    return extension === 'dcm' || 
           extension === 'dicom' || 
           mimeType === 'application/dicom' ||
           mimeType === 'image/dicom' ||
           mimeType.includes('dicom');
}