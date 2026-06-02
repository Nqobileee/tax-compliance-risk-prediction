import { createContext, useContext, useState, type ReactNode } from "react";
import type { CompanyFiling, RiskAssessment } from "../types";
import { DEFAULT_FILING } from "../types";

interface FilingContextValue {
  filing: CompanyFiling;
  setFiling: (filing: CompanyFiling) => void;
  assessment: RiskAssessment | null;
  setAssessment: (assessment: RiskAssessment | null) => void;
  reset: () => void;
}

const FilingContext = createContext<FilingContextValue | null>(null);

export function FilingProvider({ children }: { children: ReactNode }) {
  const [filing, setFiling] = useState<CompanyFiling>(DEFAULT_FILING);
  const [assessment, setAssessment] = useState<RiskAssessment | null>(null);

  const reset = () => {
    setFiling(DEFAULT_FILING);
    setAssessment(null);
  };

  return (
    <FilingContext.Provider
      value={{ filing, setFiling, assessment, setAssessment, reset }}
    >
      {children}
    </FilingContext.Provider>
  );
}

export function useFiling() {
  const ctx = useContext(FilingContext);
  if (!ctx) throw new Error("useFiling must be used within FilingProvider");
  return ctx;
}
