import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { Icon } from "./Icon";

interface NavItem {
  to: string;
  icon: string;
  label: string;
}

const NAV: NavItem[] = [
  { to: "/board", icon: "dashboard", label: "Draft Board" },
  { to: "/prospect", icon: "person_search", label: "Prospect Detail" },
  { to: "/explain", icon: "query_stats", label: "Explainability" },
  { to: "/team-fit", icon: "groups", label: "Team Fit" },
  { to: "/compare", icon: "compare_arrows", label: "Comparison" },
];

const TOP_LINKS = [
  { to: "/board", label: "Draft Board" },
  { to: "/team-fit", label: "Simulations" },
  { to: "/explain", label: "Reports" },
];

/**
 * The shared War Room shell: fixed 280px sidebar + 64px top bar, matching the
 * Apex "Front Office / Draft Command Center" design. Pages render in <Outlet/>.
 */
export function AppLayout() {
  const navigate = useNavigate();
  return (
    <div className="flex h-screen overflow-hidden bg-background text-on-background">
      {/* ---- Side nav ---- */}
      <nav className="fixed left-0 top-0 z-20 hidden h-full w-[280px] flex-col border-r border-outline-variant bg-surface py-6 md:flex">
        <div className="mb-8 px-container-padding">
          <div className="mb-2 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full border border-outline-variant bg-surface-container-highest">
              <Icon name="sports_basketball" className="text-brand-orange" size={24} />
            </div>
            <div>
              <h1 className="font-headline-md text-headline-md font-bold text-primary">
                Front Office
              </h1>
              <p className="font-label-caps text-label-caps text-on-surface-variant">
                Draft Command Center
              </p>
            </div>
          </div>
          <button
            onClick={() => navigate("/board")}
            className="mt-4 flex w-full items-center justify-center gap-2 rounded-md bg-primary-container px-4 py-2 font-label-caps text-label-caps text-on-primary transition-opacity hover:opacity-90"
          >
            <Icon name="add" size={18} />
            New Simulation
          </button>
        </div>

        <ul className="flex-1 space-y-1 overflow-y-auto px-4">
          {NAV.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                className={({ isActive }) =>
                  [
                    "flex items-center gap-3 rounded-md px-4 py-3 font-label-caps text-label-caps transition-colors",
                    isActive
                      ? "border-r-2 border-primary bg-surface-variant/30 font-bold text-primary"
                      : "text-on-surface-variant hover:bg-surface-variant",
                  ].join(" ")
                }
              >
                <Icon name={item.icon} size={22} />
                <span>{item.label}</span>
              </NavLink>
            </li>
          ))}
        </ul>

        <div className="px-4 pt-4">
          <div className="rounded-md border border-outline-variant bg-surface-container-lowest px-3 py-2">
            <p className="font-label-caps text-[10px] text-on-surface-variant">
              Demo · synthetic data
            </p>
          </div>
        </div>
      </nav>

      {/* ---- Main column ---- */}
      <div className="flex h-full w-full flex-1 flex-col md:ml-[280px] md:w-[calc(100%-280px)]">
        <header className="z-10 flex h-[64px] shrink-0 items-center justify-between border-b border-outline-variant bg-surface px-gutter">
          <div className="flex items-center gap-8">
            <span className="font-headline-md text-headline-md font-black text-on-surface md:hidden">
              DDS
            </span>
            <nav className="hidden items-center gap-6 md:flex">
              {TOP_LINKS.map((l) => (
                <NavLink
                  key={l.label}
                  to={l.to}
                  className={({ isActive }) =>
                    [
                      "font-label-caps text-label-caps transition-colors hover:text-primary",
                      isActive
                        ? "border-b-2 border-primary pb-1 font-bold text-primary"
                        : "text-on-surface-variant",
                    ].join(" ")
                  }
                >
                  {l.label}
                </NavLink>
              ))}
            </nav>
          </div>

          <div className="flex items-center gap-4">
            <div className="relative hidden sm:block">
              <span className="absolute left-3 top-1/2 -translate-y-1/2">
                <Icon name="search" size={18} className="text-on-surface-variant" />
              </span>
              <input
                className="w-48 rounded-full border border-outline-variant bg-surface-container-high py-1.5 pl-9 pr-4 font-body-sm text-body-sm text-on-surface transition-colors focus:border-primary focus:outline-none"
                placeholder="Search prospects..."
                type="text"
              />
            </div>
            <button className="flex items-center gap-2 rounded-md border border-outline-variant bg-surface-container-high px-3 py-1.5 font-label-caps text-label-caps text-on-surface hover:bg-surface-variant">
              Team Selection
              <Icon name="arrow_drop_down" size={16} />
            </button>
            <div className="ml-2 flex items-center gap-2 border-l border-outline-variant pl-4">
              <button className="text-on-surface-variant transition-colors hover:text-primary">
                <Icon name="notifications" size={22} />
              </button>
              <button className="text-on-surface-variant transition-colors hover:text-primary">
                <Icon name="settings" size={22} />
              </button>
              <div className="ml-2 flex h-8 w-8 items-center justify-center rounded-full border border-outline-variant bg-surface-container-highest">
                <Icon name="person" size={18} className="text-on-surface-variant" />
              </div>
            </div>
          </div>
        </header>

        <div className="flex items-center gap-2 border-b border-outline-variant bg-surface-container-lowest px-container-padding py-1.5">
          <Icon name="info" size={14} className="text-brand-orange" />
          <p className="font-label-caps text-[10px] text-on-surface-variant">
            2026 class · pre-draft projections — NBA outcomes not yet observed · skill mapping is
            exploratory
          </p>
        </div>
        <main className="relative flex flex-1 flex-col gap-card-gap overflow-y-auto bg-background p-container-padding">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
