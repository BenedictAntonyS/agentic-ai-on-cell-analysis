/**
 * NeuroCell Proliferation Kinetic Chart
 * Standalone High-Performance HTML5 Canvas Chart with Zero External Dependencies
 */
class KineticGrowthChart {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.data = {
      labels: ['T1 (0h)', 'T2 (4h)', 'T3 (8h)', 'T4 (12h)', 'T5 (Forecast)'],
      confluences: [18.2, 22.4, 27.8, 33.5, 39.8],
      cellCounts: [85, 96, 114, 132, 155]
    };
    this.hoverIndex = -1;
    this.setupListeners();
    this.render();
  }

  updateData(labels, confluences, cellCounts) {
    if (labels && labels.length) this.data.labels = labels;
    if (confluences && confluences.length) this.data.confluences = confluences;
    if (cellCounts && cellCounts.length) this.data.cellCounts = cellCounts;
    this.render();
  }

  setupListeners() {
    window.addEventListener('resize', () => this.render());

    this.canvas.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const count = this.data.labels.length;
      const paddingLeft = 50;
      const paddingRight = 40;
      const width = rect.width - paddingLeft - paddingRight;
      const step = width / (count - 1);

      let closestIdx = -1;
      let minDistance = 30;

      for (let i = 0; i < count; i++) {
        const px = paddingLeft + (i * step);
        const dist = Math.abs(x - px);
        if (dist < minDistance) {
          closestIdx = i;
          minDistance = dist;
        }
      }

      if (this.hoverIndex !== closestIdx) {
        this.hoverIndex = closestIdx;
        this.render();
      }
    });

    this.canvas.addEventListener('mouseleave', () => {
      if (this.hoverIndex !== -1) {
        this.hoverIndex = -1;
        this.render();
      }
    });
  }

  render() {
    if (!this.canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    const ctx = this.ctx;
    ctx.resetTransform();
    ctx.scale(dpr, dpr);

    // Clear background
    ctx.clearRect(0, 0, w, h);

    const padLeft = 60;
    const padRight = 40;
    const padTop = 30;
    const padBottom = 45;

    const chartW = w - padLeft - padRight;
    const chartH = h - padTop - padBottom;

    const labels = this.data.labels;
    const confluences = this.data.confluences;
    const maxConf = Math.max(50, Math.ceil(Math.max(...confluences) * 1.25));

    // Draw horizontal grid lines & labels
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const val = Math.round((maxConf / yTicks) * i);
      const y = padTop + chartH - ((val / maxConf) * chartH);

      ctx.beginPath();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      ctx.moveTo(padLeft, y);
      ctx.lineTo(w - padRight, y);
      ctx.stroke();

      ctx.fillText(`${val}%`, padLeft - 10, y);
    }

    // Compute coordinate points
    const points = [];
    const stepX = chartW / (labels.length - 1);
    for (let i = 0; i < labels.length; i++) {
      const x = padLeft + (i * stepX);
      const y = padTop + chartH - ((confluences[i] / maxConf) * chartH);
      points.push({ x, y });
    }

    // Gradient area beneath curve
    if (points.length > 1) {
      const grad = ctx.createLinearGradient(0, padTop, 0, padTop + chartH);
      grad.addColorStop(0, 'rgba(0, 245, 255, 0.28)');
      grad.addColorStop(0.7, 'rgba(0, 245, 255, 0.06)');
      grad.addColorStop(1, 'rgba(0, 245, 255, 0.0)');

      ctx.beginPath();
      ctx.moveTo(points[0].x, padTop + chartH);
      for (let i = 0; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
      }
      ctx.lineTo(points[points.length - 1].x, padTop + chartH);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();
    }

    // Draw solid historical line (T1 to T4)
    if (points.length >= 4) {
      ctx.beginPath();
      ctx.strokeStyle = '#00f5ff';
      ctx.lineWidth = 3;
      ctx.shadowColor = 'rgba(0, 245, 255, 0.7)';
      ctx.shadowBlur = 12;

      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i <= 3; i++) {
        ctx.lineTo(points[i].x, points[i].y);
      }
      ctx.stroke();

      // Draw dashed forecast segment (T4 to T5)
      ctx.beginPath();
      ctx.strokeStyle = '#a855f7';
      ctx.shadowColor = 'rgba(168, 85, 247, 0.8)';
      ctx.shadowBlur = 14;
      ctx.lineWidth = 3;
      ctx.setLineDash([6, 5]);
      ctx.moveTo(points[3].x, points[3].y);
      ctx.lineTo(points[4].x, points[4].y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.shadowBlur = 0;
    }

    // Draw Points & Labels
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const isForecast = (i === points.length - 1);

      // Outer halo
      ctx.beginPath();
      ctx.arc(p.x, p.y, isForecast ? 8 : 6, 0, Math.PI * 2);
      ctx.fillStyle = isForecast ? 'rgba(168, 85, 247, 0.25)' : 'rgba(0, 245, 255, 0.2)';
      ctx.fill();

      // Point circle
      ctx.beginPath();
      ctx.arc(p.x, p.y, isForecast ? 5 : 4, 0, Math.PI * 2);
      ctx.fillStyle = isForecast ? '#a855f7' : '#00f5ff';
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.fill();
      ctx.stroke();

      // Label below X-axis
      ctx.fillStyle = (this.hoverIndex === i) ? '#00f5ff' : '#94a3b8';
      ctx.font = (this.hoverIndex === i) ? 'bold 11px "Inter", sans-serif' : '11px "Inter", sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(labels[i], p.x, padTop + chartH + 12);
    }

    // Draw Hover Tooltip if hovering
    if (this.hoverIndex >= 0 && this.hoverIndex < points.length) {
      const idx = this.hoverIndex;
      const p = points[idx];
      const conf = confluences[idx];
      const cells = this.data.cellCounts[idx] || 'N/A';
      const label = labels[idx];

      const tipText1 = `${label}: ${conf}% Confluence`;
      const tipText2 = `Detected: ${cells} cell bodies`;

      ctx.font = 'bold 11px "Inter", sans-serif';
      const tw = Math.max(ctx.measureText(tipText1).width, ctx.measureText(tipText2).width) + 24;
      const th = 48;

      let tx = p.x - (tw / 2);
      let ty = p.y - th - 12;

      // Bound inside canvas
      if (tx < 10) tx = 10;
      if (tx + tw > w - 10) tx = w - tw - 10;
      if (ty < 5) ty = p.y + 16;

      // Tooltip background
      ctx.fillStyle = 'rgba(7, 10, 18, 0.92)';
      ctx.strokeStyle = idx === points.length - 1 ? '#a855f7' : '#00f5ff';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(tx, ty, tw, th, 8) : ctx.rect(tx, ty, tw, th);
      ctx.fill();
      ctx.stroke();

      // Tooltip texts
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillStyle = '#fff';
      ctx.fillText(tipText1, tx + 12, ty + 8);
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillStyle = '#94a3b8';
      ctx.fillText(tipText2, tx + 12, ty + 26);
    }
  }
}

window.KineticGrowthChart = KineticGrowthChart;
