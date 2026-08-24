import Link from "next/link";
import { AppHeader } from "@/components/AppHeader";
import { Icon } from "@/components/icons";
import { SiteFooter } from "@/components/SiteFooter";

const analyses = [
  {
    icon: "count" as const,
    title: "Event counts",
    body: "Analyse counts for individual cells and compare candidate population models.",
    href: "/workspace?kind=event-counts",
  },
  {
    icon: "user" as const,
    title: "Donor aware counts",
    body: "Separate variation within donors from differences between donors.",
    href: "/workspace?kind=event-counts",
  },
  {
    icon: "trajectory" as const,
    title: "Contact trajectories",
    body: "Test whether previous contacts change later killing decisions.",
    href: "/workspace?kind=trajectory",
  },
];

export default function HomePage() {
  return (
    <div className="marketingPage">
      <AppHeader />
      <main>
        <section className="landingHero conciseHero" id="how-it-works">
          <div className="heroCopy">
            <div className="kicker"><span /> Bayesian Analysis Resolving Randomness and Alternative Causes Underlying Differential Activity</div>
            <h1>BARRACUDA</h1>
            <p className="heroLead">
              Bayesian inference for understanding variation in immune cell cytotoxicity.
            </p>
            <div className="heroActions">
              <Link className="button buttonPrimary" href="/workspace">
                Start an analysis <Icon name="arrow" />
              </Link>
            </div>
            <p className="guestNote">
              <Icon name="lock" /> No account required. Sign in only if you want to save your work.
            </p>
          </div>
        </section>

        <section className="analysisChooser" id="analyses">
          <div className="compactSectionHeading">
            <span className="sectionLabel">Choose an analysis</span>
            <h2>Choose an analysis.</h2>
          </div>
          <div className="analysisChoiceGrid">
            {analyses.map((analysis) => (
              <Link className="analysisChoiceCard" href={analysis.href} key={analysis.title}>
                <Icon name={analysis.icon} />
                <h3>{analysis.title}</h3>
                <p>{analysis.body}</p>
                <span>Open workspace <Icon name="arrow" /></span>
              </Link>
            ))}
          </div>
        </section>

        <section className="compactResearch" id="research">
          <div>
            <span className="sectionLabel light">Research team and code</span>
            <p>
              Elephes Sung, Cathal Hosty, Leanne Peiser, Lara Stepan, Daniel M Davis and Ruben Perez-Carrasco.
            </p>
          </div>
          <a
            className="button buttonLight"
            href="https://github.com/sthsci/Barracuda"
            target="_blank"
            rel="noreferrer"
          >
            View research code <Icon name="arrow" />
          </a>
        </section>
      </main>
      <SiteFooter />
    </div>
  );
}
