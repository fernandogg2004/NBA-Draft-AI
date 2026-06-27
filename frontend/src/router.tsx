import { createBrowserRouter, Navigate } from "react-router-dom";
import { AppLayout } from "./components/AppLayout";
import { DraftBoard } from "./pages/DraftBoard";
import { ProspectDetail } from "./pages/ProspectDetail";
import { Explainability } from "./pages/Explainability";
import { TeamFit } from "./pages/TeamFit";
import { Comparison } from "./pages/Comparison";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppLayout />,
    children: [
      { index: true, element: <Navigate to="/board" replace /> },
      { path: "board", element: <DraftBoard /> },
      { path: "prospect/:playerId", element: <ProspectDetail /> },
      { path: "prospect", element: <ProspectDetail /> },
      { path: "explain/:playerId", element: <Explainability /> },
      { path: "explain", element: <Explainability /> },
      { path: "team-fit", element: <TeamFit /> },
      { path: "compare", element: <Comparison /> },
      { path: "*", element: <Navigate to="/board" replace /> },
    ],
  },
]);
