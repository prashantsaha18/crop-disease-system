// result_screen.dart
// ────────────────────────────────────────────────────────────────────────────
// Screen 2 – Analysis Result
//
// Shows:
//   • Original leaf photo + GradCAM heatmap overlay (toggle)
//   • Predicted disease name + confidence arc indicator
//   • Top-5 predictions list
//   • Treatment recommendation card
//   • Inference time badge
//   • Re-scan button
// ────────────────────────────────────────────────────────────────────────────

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image/image.dart' as img;
import 'package:percent_indicator/circular_percent_indicator.dart';

import '../utils/classifier.dart';

// ── args ─────────────────────────────────────────────────────────────────── //

class ResultScreenArgs {
  final File                imageFile;
  final ClassificationResult result;
  const ResultScreenArgs({required this.imageFile, required this.result});
}

// ── screen ───────────────────────────────────────────────────────────────── //

class ResultScreen extends StatefulWidget {
  final ResultScreenArgs args;
  const ResultScreen({super.key, required this.args});

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  bool _showOverlay = true;
  Uint8List? _overlayBytes;
  Uint8List? _originalBytes;

  @override
  void initState() {
    super.initState();
    _prepareImages();
  }

  Future<void> _prepareImages() async {
    final bytes = await widget.args.imageFile.readAsBytes();
    setState(() => _originalBytes = bytes);

    final overlay = widget.args.result.heatmapOverlay;
    if (overlay != null) {
      final encoded = img.encodeJpg(overlay, quality: 92);
      setState(() => _overlayBytes = Uint8List.fromList(encoded));
    }
  }

  // ── helpers ───────────────────────────────────────────────────────────── //

  bool get _isHealthy =>
      widget.args.result.label.toLowerCase().contains('healthy');

  Color _confidenceColor(double c) {
    if (c >= 0.85) return const Color(0xFF1A472A);
    if (c >= 0.65) return const Color(0xFFE8A020);
    return const Color(0xFFD64045);
  }

  String _severityLabel(double c) {
    if (_isHealthy)  return 'Healthy Plant';
    if (c >= 0.85)   return 'High Confidence';
    if (c >= 0.65)   return 'Moderate Confidence';
    return 'Low Confidence';
  }

  String _treatmentText() {
    return kTreatmentDB[widget.args.result.label] ??
        'Consult a local agronomist for a specific treatment plan.';
  }

  // ── build ─────────────────────────────────────────────────────────────── //

  @override
  Widget build(BuildContext context) {
    final r  = widget.args.result;
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      backgroundColor: cs.background,
      body: CustomScrollView(
        physics: const BouncingScrollPhysics(),
        slivers: [
          _buildAppBar(context, cs, r),
          SliverPadding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            sliver: SliverList(
              delegate: SliverChildListDelegate([
                const SizedBox(height: 16),
                _buildImageCard(cs),
                const SizedBox(height: 20),
                _buildDiseaseCard(cs, r),
                const SizedBox(height: 16),
                _buildTreatmentCard(cs, r),
                const SizedBox(height: 16),
                _buildTop5Card(cs, r),
                const SizedBox(height: 16),
                _buildMetaChips(cs, r),
                const SizedBox(height: 32),
                _buildRescanButton(context, cs),
                const SizedBox(height: 40),
              ]),
            ),
          ),
        ],
      ),
    );
  }

  // ── app bar ───────────────────────────────────────────────────────────── //

  Widget _buildAppBar(BuildContext context, ColorScheme cs, ClassificationResult r) {
    return SliverAppBar(
      expandedHeight: 0,
      pinned: true,
      backgroundColor: cs.background,
      leading: IconButton(
        icon: Container(
          decoration: BoxDecoration(
            color: cs.surfaceVariant,
            borderRadius: BorderRadius.circular(10),
          ),
          padding: const EdgeInsets.all(6),
          child: Icon(Icons.arrow_back_ios_new_rounded, size: 16, color: cs.primary),
        ),
        onPressed: () => Navigator.pop(context),
      ),
      title: Text(
        'Analysis Result',
        style: GoogleFonts.sora(
          fontSize: 18, fontWeight: FontWeight.w700,
          color: const Color(0xFF1A472A),
        ),
      ),
      actions: [
        Padding(
          padding: const EdgeInsets.only(right: 16),
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            decoration: BoxDecoration(
              color: const Color(0xFF1A472A).withOpacity(0.08),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.wifi_off_rounded, size: 12, color: cs.primary),
                const SizedBox(width: 4),
                Text(
                  'Offline',
                  style: GoogleFonts.sora(
                    fontSize: 11, fontWeight: FontWeight.w600, color: cs.primary,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  // ── image card ────────────────────────────────────────────────────────── //

  Widget _buildImageCard(ColorScheme cs) {
    final hasOverlay = _overlayBytes != null;

    return Card(
      child: Column(
        children: [
          // image
          ClipRRect(
            borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
            child: AspectRatio(
              aspectRatio: 1,
              child: AnimatedSwitcher(
                duration: const Duration(milliseconds: 350),
                child: (_showOverlay && hasOverlay)
                    ? Image.memory(
                        _overlayBytes!,
                        key: const ValueKey('overlay'),
                        fit: BoxFit.cover,
                        width: double.infinity,
                      )
                    : (_originalBytes != null
                        ? Image.memory(
                            _originalBytes!,
                            key: const ValueKey('original'),
                            fit: BoxFit.cover,
                            width: double.infinity,
                          )
                        : Container(
                            color: cs.surfaceVariant,
                            child: Center(
                              child: CircularProgressIndicator(color: cs.primary),
                            ),
                          )),
              ),
            ),
          ),
          // toggle strip
          if (hasOverlay)
            Container(
              decoration: BoxDecoration(
                color: cs.surfaceVariant,
                borderRadius: const BorderRadius.vertical(bottom: Radius.circular(20)),
              ),
              child: Row(
                children: [
                  _imgToggleBtn('Original', false, cs),
                  _imgToggleBtn('GradCAM Heatmap', true, cs),
                ],
              ),
            ),
        ],
      ),
    ).animate().fadeIn(duration: 400.ms).slideY(begin: 0.08, end: 0);
  }

  Widget _imgToggleBtn(String label, bool isOverlay, ColorScheme cs) {
    final selected = _showOverlay == isOverlay;
    return Expanded(
      child: GestureDetector(
        onTap: () => setState(() => _showOverlay = isOverlay),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          padding: const EdgeInsets.symmetric(vertical: 11),
          decoration: BoxDecoration(
            color: selected ? cs.primary : Colors.transparent,
            borderRadius: BorderRadius.only(
              bottomLeft: isOverlay ? Radius.zero : const Radius.circular(20),
              bottomRight: isOverlay ? const Radius.circular(20) : Radius.zero,
            ),
          ),
          child: Text(
            label,
            textAlign: TextAlign.center,
            style: GoogleFonts.sora(
              fontSize: 12, fontWeight: FontWeight.w600,
              color: selected ? Colors.white : cs.onSurfaceVariant,
            ),
          ),
        ),
      ),
    );
  }

  // ── disease card ──────────────────────────────────────────────────────── //

  Widget _buildDiseaseCard(ColorScheme cs, ClassificationResult r) {
    final conf      = r.confidence;
    final confColor = _confidenceColor(conf);
    final severity  = _severityLabel(conf);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // confidence arc
            CircularPercentIndicator(
              radius: 48,
              lineWidth: 7,
              percent: conf.clamp(0.0, 1.0),
              animation: true,
              animationDuration: 1000,
              progressColor: confColor,
              backgroundColor: confColor.withOpacity(0.12),
              circularStrokeCap: CircularStrokeCap.round,
              center: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    '${(conf * 100).toStringAsFixed(0)}%',
                    style: GoogleFonts.sora(
                      fontSize: 16, fontWeight: FontWeight.w700,
                      color: confColor,
                    ),
                  ),
                  Text(
                    'conf.',
                    style: GoogleFonts.sora(
                      fontSize: 9, color: const Color(0xFF6B7860),
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(width: 16),

            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        width: 8, height: 8,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: _isHealthy
                              ? const Color(0xFF2E7D32)
                              : const Color(0xFFD64045),
                        ),
                      ),
                      const SizedBox(width: 6),
                      Text(
                        _isHealthy ? 'No disease found' : 'Disease detected',
                        style: GoogleFonts.sora(
                          fontSize: 11, fontWeight: FontWeight.w600,
                          color: _isHealthy
                              ? const Color(0xFF2E7D32)
                              : const Color(0xFFD64045),
                          letterSpacing: 0.5,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  Text(
                    r.displayName,
                    style: GoogleFonts.sora(
                      fontSize: 17, fontWeight: FontWeight.w700,
                      color: const Color(0xFF1A472A),
                      height: 1.25,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
                    decoration: BoxDecoration(
                      color: confColor.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: Text(
                      severity,
                      style: GoogleFonts.sora(
                        fontSize: 11, fontWeight: FontWeight.w600,
                        color: confColor,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    ).animate().fadeIn(delay: 150.ms, duration: 400.ms).slideY(begin: 0.1, end: 0);
  }

  // ── treatment card ────────────────────────────────────────────────────── //

  Widget _buildTreatmentCard(ColorScheme cs, ClassificationResult r) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 36, height: 36,
                  decoration: BoxDecoration(
                    color: const Color(0xFFE8A020).withOpacity(0.15),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: const Icon(
                    Icons.healing_rounded,
                    color: Color(0xFFE8A020),
                    size: 20,
                  ),
                ),
                const SizedBox(width: 12),
                Text(
                  'Treatment Recommendation',
                  style: GoogleFonts.sora(
                    fontSize: 15, fontWeight: FontWeight.w600,
                    color: const Color(0xFF1A472A),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 14),
            const Divider(),
            const SizedBox(height: 12),
            Text(
              _treatmentText(),
              style: GoogleFonts.sora(
                fontSize: 13.5,
                color: const Color(0xFF2D3A27),
                height: 1.7,
              ),
            ),
            const SizedBox(height: 14),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: const Color(0xFFF4F7F2),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: const Color(0xFFDEE5D8)),
              ),
              child: Row(
                children: [
                  const Icon(Icons.info_outline_rounded, size: 16,
                      color: Color(0xFF6B7860)),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Always verify with a certified agronomist before applying treatments.',
                      style: GoogleFonts.sora(
                        fontSize: 11, color: const Color(0xFF6B7860), height: 1.5,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    ).animate().fadeIn(delay: 250.ms, duration: 400.ms).slideY(begin: 0.1, end: 0);
  }

  // ── top-5 card ────────────────────────────────────────────────────────── //

  Widget _buildTop5Card(ColorScheme cs, ClassificationResult r) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Top Predictions',
              style: GoogleFonts.sora(
                fontSize: 15, fontWeight: FontWeight.w600,
                color: const Color(0xFF1A472A),
              ),
            ),
            const SizedBox(height: 12),
            ...r.top5.asMap().entries.map((e) {
              final rank = e.key;
              final pred = e.value;
              final isTop = rank == 0;
              return Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: Row(
                  children: [
                    // rank badge
                    Container(
                      width: 24, height: 24,
                      decoration: BoxDecoration(
                        color: isTop
                            ? cs.primary
                            : cs.surfaceVariant,
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Center(
                        child: Text(
                          '${rank + 1}',
                          style: GoogleFonts.sora(
                            fontSize: 11, fontWeight: FontWeight.w700,
                            color: isTop ? Colors.white : const Color(0xFF6B7860),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            pred.displayName,
                            style: GoogleFonts.sora(
                              fontSize: 12.5,
                              fontWeight: isTop ? FontWeight.w600 : FontWeight.w400,
                              color: const Color(0xFF1C2117),
                            ),
                          ),
                          const SizedBox(height: 4),
                          ClipRRect(
                            borderRadius: BorderRadius.circular(4),
                            child: LinearProgressIndicator(
                              value: pred.confidence,
                              minHeight: 5,
                              backgroundColor: cs.surfaceVariant,
                              color: isTop
                                  ? cs.primary
                                  : cs.primary.withOpacity(0.35),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 10),
                    Text(
                      '${(pred.confidence * 100).toStringAsFixed(1)}%',
                      style: GoogleFonts.sora(
                        fontSize: 12, fontWeight: FontWeight.w600,
                        color: isTop ? cs.primary : const Color(0xFF6B7860),
                      ),
                    ),
                  ],
                ),
              );
            }),
          ],
        ),
      ),
    ).animate().fadeIn(delay: 320.ms, duration: 400.ms).slideY(begin: 0.1, end: 0);
  }

  // ── meta chips ────────────────────────────────────────────────────────── //

  Widget _buildMetaChips(ColorScheme cs, ClassificationResult r) {
    return Wrap(
      spacing: 8, runSpacing: 8,
      children: [
        _MetaChip(
          icon: Icons.timer_rounded,
          label: '${r.inferenceTime.inMilliseconds} ms inference',
          cs: cs,
        ),
        _MetaChip(
          icon: Icons.memory_rounded,
          label: 'EfficientNetV2-S',
          cs: cs,
        ),
        _MetaChip(
          icon: Icons.wifi_off_rounded,
          label: 'No internet used',
          cs: cs,
        ),
      ],
    ).animate().fadeIn(delay: 400.ms, duration: 400.ms);
  }

  // ── rescan button ─────────────────────────────────────────────────────── //

  Widget _buildRescanButton(BuildContext context, ColorScheme cs) {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton.icon(
        onPressed: () => Navigator.pop(context),
        icon: const Icon(Icons.camera_alt_rounded, size: 20),
        label: const Text('Scan Another Leaf'),
      ),
    ).animate().fadeIn(delay: 450.ms, duration: 400.ms);
  }
}

// ── helper widget ─────────────────────────────────────────────────────────── //

class _MetaChip extends StatelessWidget {
  final IconData   icon;
  final String     label;
  final ColorScheme cs;

  const _MetaChip({required this.icon, required this.label, required this.cs});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: cs.surfaceVariant,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 13, color: cs.primary),
          const SizedBox(width: 5),
          Text(
            label,
            style: GoogleFonts.sora(
              fontSize: 11.5, fontWeight: FontWeight.w500,
              color: const Color(0xFF2D3A27),
            ),
          ),
        ],
      ),
    );
  }
}
