import './App.css';
import { handleFolderSelect } from './SendFiles';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <div className='container'>
          <div className='main-title-box'>
            <span className='main-title'>
              Загрузите .dcm файлы КТ печени по кнопке ниже, чтобы создать 3D-модель
            </span>
          </div>
        </div>
      </header>
      <main>
        <div className='container'>
          <div id='model' className='model-space'>
            
          </div>
          <div className='tool-btns'>
            <form id="upload-box" method="POST">
                <div>
                  <input id="upload-files" type="file" webkitdirectory="true" directory="true" multiple onChange={(e) => handleFolderSelect(e)}></input>
                  <label htmlFor="upload-files">Выберите папку</label>
                </div>
            </form>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
