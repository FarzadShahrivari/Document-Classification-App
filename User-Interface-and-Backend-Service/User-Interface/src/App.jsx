/*
    Import CSS file for styling and necessary React hooks and axios library for managing state and making HTTP requests.
*/
import './App.css'
import { useState, useEffect } from "react";
import axios from "axios";

function App() {
    // State hooks for managing data and user input
    // eslint-disable-next-line
    const [data, setData] = useState("");
    const [val, setVal] = useState("Upload image to predict");

    // State hooks for managing file upload
    const [filename, setFilename] = useState("No file Uploaded")

    // Fetch initial data from backend upon component mount
    useEffect(() => {
        fetch("http://localhost:5000")
            .then((res) => res.json())
            .then((data) => {
                console.log(data);
                setData(data.message);
            });
    }, []);
    const [file, setFile] = useState(null);

    // Handle form submission
    const handleSubmit = async (event) => {
        event.preventDefault();

        // Create FormData object to send file to backend
        const formData = new FormData();
        formData.append("file", file);

        try {
            axios.post("http://localhost:5000/upload", formData).then((res) => {
                console.log(res.data.message);
                setVal(res.data.message);
            });
            alert("File uploaded successfully");
        } catch (error) {
            console.error(error);
        }

    };

    // Handle file input change
    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        setFilename(file.name);
    };

    // Building the app
    return (
        <>
            {/* Header section with title for the ID Card Classification app */}
            <h1 className="animate-fade-in-with-shadow" style={{ fontSize: '6rem', fontWeight: '700', marginBottom: '4rem', color: '#7FD7B8' }}>
                ID Card Classification
            </h1>
            {/* Putting next line with textShadow */}
            <p style={{ fontSize: '1.3rem', color: 'black', fontWeight: 'normal', marginBottom: '0.1rem', lineHeight: '1.5', textShadow: '0 0 5px rgba(0, 0, 0, 0.3)' }}>
                Upload the image file to detect
            </p>
            {/* File upload button with a blue cloud-like logo */}
            <form onSubmit={handleSubmit}>

                <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '3rem', padding: '1.5rem' }}>
                    <label style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', cursor: 'pointer' }}>
                        <span style={{ fontSize: '1rem', padding: '1.7rem', backgroundColor: '#7FD7B8', color: '#FFFFFF', textShadow: '0 0 5px rgba(0, 0, 0, 0.3)', borderRadius: '1rem'}}>Select a file</span>
                        <input type="file" name="file" onChange={(e) => { setFile(e.target.files[0]); handleFileUpload(e) }} style={{ display: 'none' }} />
                        <svg
                            fill="blue"
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 20 20"
                            style={{ width: '30px', height: '30px', marginTop: '-75px' }}
                        >
                            <path d="M16.88 9.1A4 4 0 0 1 16 17H5a5 5 0 0 1-1-9.9V7a3 3 0 0 1 4.52-2.59A4.98 4.98 0 0 1 17 8c0 .38-.04.74-.12 1.1zM11 11h3l-4-4-4 4h3v3h2v-3z" />
                        </svg>
                    </label>
                </div>

                {/* Display the name of the uploaded file */}
                <span style={{ fontSize: '1.3rem', color: 'black', marginBottom: '3rem', fontWeight: 'normal', lineHeight: '1.5', textShadow: '0 0 5px rgba(0, 0, 0, 0.3)' }}>File Uploaded: {filename}</span>

                {/* Button for triggering file upload */}
                <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '0.5rem', padding: '1.5rem' }}>
                    <button style={{ fontSize: '1.1rem', padding: '1rem', backgroundColor: '#7FD7B8', color: '#FFFFFF', borderRadius: '1rem', cursor: 'pointer', border: 'none', outline: 'none' }} type="submit">
                        PREDICT
                    </button>
                </div>
            </form>

            {/* Display the predicted class of the uploaded image */}
            <div>
                <span style={{ fontSize: '1.3rem', color: 'black', fontWeight: 'normal', lineHeight: '1.5', textShadow: '0 0 5px rgba(0, 0, 0, 0.3)' }}>
                    Detected Image is : {val}
                </span>
            </div>
        </>
    );
}

export default App;