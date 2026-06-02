import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { FilingProvider } from "./context/FilingContext";
import { Home } from "./pages/Home";
import { InputForm } from "./pages/InputForm";
import { Results } from "./pages/Results";
import { Stats } from "./pages/Stats";

export default function App() {
  return (
    <BrowserRouter>
      <FilingProvider>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Home />} />
            <Route path="assess" element={<InputForm />} />
            <Route path="results" element={<Results />} />
            <Route path="stats" element={<Stats />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </FilingProvider>
    </BrowserRouter>
  );
}
