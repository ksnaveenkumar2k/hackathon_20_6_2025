import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from '../../frontend/src/Pages/home';
import About from '../../frontend/src/Pages/about';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;