import { Link, useLocation } from "react-router-dom"
import { Home, Info, Target } from 'lucide-react'

const Navbar = () => {
  const location = useLocation()

  const navItems = [
    { path: "/", label: "Home", icon: Home },
    { path: "/about", label: "About", icon: Info },
  ]

  return (
    <nav className="bg-black shadow-lg sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex items-center space-x-2">
            <Target className="h-8 w-8 text-yellow-400" />
            <span className="text-white text-xl font-bold">Reality Matcher</span>
          </div>

          {/* Navigation Links */}
          <div className="flex space-x-8">
            {navItems.map(({ path, label, icon: Icon }) => (
              <Link
                key={path}
                to={path}
                className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                  location.pathname === path
                    ? "bg-yellow-400 text-black"
                    : "text-white hover:bg-yellow-400 hover:text-black"
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{label}</span>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
