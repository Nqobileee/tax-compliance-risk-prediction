import { NavLink } from "react-router-dom";
import "./Navbar.css";

const links = [
  { to: "/", label: "Home" },
  { to: "/assess", label: "Assess Filing" },
  { to: "/stats", label: "System Stats" },
];

export function Navbar() {
  return (
    <header className="nav-shell">
      <nav className="nav-bar">
        <NavLink to="/" className="nav-brand">
          <span className="nav-logo">TG</span>
          <div>
            <span className="nav-title">TaxGuard</span>
            <span className="nav-sub">ZIMRA Risk Intelligence</span>
          </div>
        </NavLink>

        <ul className="nav-links">
          {links.map(({ to, label }) => (
            <li key={to}>
              <NavLink
                to={to}
                className={({ isActive }) =>
                  isActive ? "nav-link active" : "nav-link"
                }
                end={to === "/"}
              >
                {label}
              </NavLink>
            </li>
          ))}
        </ul>

        <NavLink to="/assess" className="nav-cta">
          Run Assessment
        </NavLink>
      </nav>
    </header>
  );
}
