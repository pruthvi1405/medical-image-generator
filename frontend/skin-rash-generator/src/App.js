import React, { useState } from 'react';
import './App.css';

function App() {
  const [skinRashType, setSkinRashType] = useState('');
  const [skinColor, setSkinColor] = useState('');
  const [bodyPart, setBodyPart] = useState('');
  const [generatedImage, setGeneratedImage] = useState(null);

  const handleGenerate = async () => {
    try {
      const response = await fetch('https://2a25-35-247-81-65.ngrok-free.app/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          skinRashType,
          skinColor,
          bodyPart,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate image');
      }

      const data = await response.json();
      setGeneratedImage(data.image);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="App">
      <h1>Skin Rash Image Generator</h1>

      <div>
        <label>Type of Skin Rash:</label>
        <select value={skinRashType} onChange={(e) => setSkinRashType(e.target.value)}>
          <option value="">Select a rash type</option>
          <option value="eczema">Eczema</option>
          <option value="ringworm">Ringworm</option>
          <option value="dermatitis">Dermatitis</option>
        </select>
      </div>

      <div>
        <label>Skin Color:</label>
        <select value={skinColor} onChange={(e) => setSkinColor(e.target.value)}>
          <option value="">Select a skin color</option>
          <option value="fair">Fair</option>
          <option value="brown">Brown</option>
          <option value="dark">Dark</option>
          {/* Add more options as needed */}
        </select>
      </div>

      <div>
        <label>Body Part:</label>
        <select value={bodyPart} onChange={(e) => setBodyPart(e.target.value)}>
          <option value="">Select a body part</option>
          <option value="arm">Arm</option>
          <option value="leg">Leg</option>
          <option value="face">Face</option>
          {/* Add more options as needed */}
        </select>
      </div>

      <button onClick={handleGenerate}>Generate Image</button>

      {generatedImage && (
        <div>
          <h2>Generated Image:</h2>
          <img src={`data:image/png;base64,${generatedImage}`} alt="Generated Skin Rash" />
        </div>
      )}
    </div>
  );
}

export default App;

