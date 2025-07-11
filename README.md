# AI Package Delivery Optimizer 

Developed by:  
**Leen Alqazaqi **  
**Duha Imad **

---

##  Overview

This project solves the package-to-vehicle assignment and route optimization problem using **two AI algorithms**:  
- Genetic Algorithm  
- Simulated Annealing  

We built it as a **web application using Flask** to provide an interactive user interface for uploading data and viewing optimized results in the browser.

A detailed explanation of the project is available in the attached **PDF report**.

---

##  How to Run the Project

To run the Flask website:

### 1. Folder Structure

 project folder should look like this:

```
project/
├── app.py                   # Flask server file
├── templates/
│   └── [ .html files]
└── static/
    └── uploads/             # Folder to store uploaded files
```

### 2. Setup and Run

1. Make sure you have Python and Flask installed:
   ```bash
   pip install flask
   ```

2. Run the server:
   ```bash
   python app.py
   ```

3. Open your browser and go to [http://localhost:5000](http://localhost:5000)

---

## 📄 Notes

- Place your `.html` files inside the `templates/` folder.
- Create a folder named `uploads` inside `static/` for file handling.
- The server file (`app.py`) must be placed **at the root of the project**, not inside any folder.

---

## Acknowledgment

Thanks for checking out our project!



