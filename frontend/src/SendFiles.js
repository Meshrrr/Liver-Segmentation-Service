import { showModel } from "./Show3DModel";

const API_URL = 'http://localhost:8000';

export function handleFolderSelect(event) {

    let selectedFiles = Array.from(event.target.files);
    console.log(`Выбрано файлов: ${selectedFiles.length}`);

    if (selectedFiles.length === 0) {
        alert('Выберите папку с файлами');
        return
    }

    sendFiles(selectedFiles)
}


async function sendFiles(files) {
    console.log('Отправка файлов на сервер...');

    const dcmFiles = files.filter(file => isDcmFile(file));

    if (dcmFiles.length === 0) {
        alert('В папке нет DICOM файлов (.dcm)!');
        return
    }

    try {
        // Загружаем первый файл для проверки
        const testFile = dcmFiles[0];
        const formData = new FormData();
        formData.append('file', testFile);

        console.log('Загрузка файла...');
        const uploadResponse = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) {
            throw new Error('Ошибка загрузки файла');
        }

        const uploadData = await uploadResponse.json();
        console.log('Файл загружен:', uploadData);
        const fileId = uploadData.file_id;

        // Шаг 2: Получаем метаданные
        console.log('Получение метаданных...');
        const metadataResponse = await fetch(`${API_URL}/dicom/${fileId}/metadata`);
        const metadata = await metadataResponse.json();
        console.log('Метаданные:', metadata);

        console.log('Получение срезов...');
        const slices = [];

        const previewResponse = await fetch(
            `${API_URL}/dicom/${fileId}/preview?window_center=40&window_width=400`
        );

        if (previewResponse.ok) {
            const previewData = await previewResponse.json();
            console.log('Превью получено');

            if (previewData.image) {
                const img = new Image();
                img.src = previewData.image;
                await new Promise(resolve => img.onload = resolve);
                slices.push(img);
            }
        }

        if (slices.length > 0) {
            showModel(slices);
            console.log('Модель отображена');
        } else {
            showModel(dcmFiles.slice(0, 50));
        }

        alert('Файлы загружены! Модель построена.');

    } catch (error) {
        console.error('Ошибка:', error);
        showModel(dcmFiles.slice(0, 50));
        alert('Сервер недоступен, показаны локальные файлы');
    }

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