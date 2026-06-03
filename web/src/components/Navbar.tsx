import { useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import "./Navbar.css";

const links = [
  { to: "/", label: "Home" },
  { to: "/assess", label: "Assess Filing" },
  { to: "/stats", label: "System Stats" },
];

export function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setMenuOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    document.body.style.overflow = menuOpen ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [menuOpen]);

  return (
    <header className="nav-shell">
      <nav className="nav-bar" aria-label="Main navigation">
        <NavLink to="/" className="nav-brand" onClick={() => setMenuOpen(false)}>
          <span className="nav-logo">TG</span>
          <div className="nav-brand-text">
            <span className="nav-title">TaxGuard</span>
            <span className="nav-sub">ZIMRA Risk Intelligence</span>
          </div>
        </NavLink>

        <button
          type="button"
          className="nav-toggle"
          aria-expanded={menuOpen}
          aria-controls="nav-menu"
          aria-label={menuOpen ? "Close menu" : "Open menu"}
          onClick={() => setMenuOpen((o) => !o)}
        >
          <span className={`nav-toggle-bar ${menuOpen ? "open" : ""}`} />
          <span className={`nav-toggle-bar ${menuOpen ? "open" : ""}`} />
          <span className={`nav-toggle-bar ${menuOpen ? "open" : ""}`} />
        </button>

        <div
          id="nav-menu"
          className={`nav-panel ${menuOpen ? "nav-panel-open" : ""}`}
        >
          <ul className="nav-links">
            {links.map(({ to, label }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  className={({ isActive }) =>
                    isActive ? "nav-link active" : "nav-link"
                  }
                  end={to === "/"}
                  onClick={() => setMenuOpen(false)}
                >
                  {label}
                </NavLink>
              </li>
            ))}
          </ul>

          <NavLink
            to="/assess"
            className="nav-cta"
            onClick={() => setMenuOpen(false)}
          >
            Run Assessment
          </NavLink>
        </div>
      </nav>

      {menuOpen && (
        <button
          type="button"
          className="nav-backdrop"
          aria-label="Close menu"
          onClick={() => setMenuOpen(false)}
        />
      )}
    </header>
  );
}
