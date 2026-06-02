// home_screen.dart
// ────────────────────────────────────────────────────────────────────────────
// Screen 1 – Landing / capture screen
//
// Features:
//   • Animated hero logo with botanical motif
//   • Camera capture button (full-res photo)
//   • Gallery picker button
//   • Animated scanning overlay while model runs inference
//   • Soft error snackbar on failure
// ────────────────────────────────────────────────────────────────────────────

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';

import '../utils/classifier.dart';
import 'result_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen>
    with SingleTickerProviderStateMixin {

  final ImagePicker _picker = ImagePicker();
  bool _isAnalysing = false;
  late AnimationController _scanController;
  late Animation<double>   _scanAnim;

  @override
  void initState() {
    super.initState();
    _scanController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);
    _scanAnim = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: _scanController, curve: Curves.easeInOut),
    );

    // Pre-warm the classifier
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      await CropClassifier.load();
    });
  }

  @override
  void dispose() {
    _scanController.dispose();
    super.dispose();
  }

  // ── image capture ─────────────────────────────────────────────────────── //

  Future<void> _capture(ImageSource source) async {
    if (_isAnalysing) return;
    try {
      final picked = await _picker.pickImage(
        source: source,
        imageQuality: 92,
        maxWidth: 1200,
        maxHeight: 1200,
      );
      if (picked == null) return;
      await _runInference(File(picked.path));
    } catch (e) {
      _showError('Could not access ${ source == ImageSource.camera ? 'camera' : 'gallery'}');
    }
  }

  Future<void> _runInference(File imageFile) async {
    setState(() => _isAnalysing = true);
    try {
      final clf    = await CropClassifier.load();
      final result = await clf.classify(imageFile);

      if (!mounted) return;
      Navigator.pushNamed(
        context,
        '/result',
        arguments: ResultScreenArgs(
          imageFile: imageFile,
          result:    result,
        ),
      );
    } catch (e) {
      _showError('Analysis failed. Please try a clearer image.');
    } finally {
      if (mounted) setState(() => _isAnalysing = false);
    }
  }

  void _showError(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg)),
    );
  }

  // ── build ─────────────────────────────────────────────────────────────── //

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final cs    = theme.colorScheme;

    return Scaffold(
      backgroundColor: cs.background,
      body: Stack(
        children: [
          _BackgroundDecoration(),

          SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 20),
                  _buildAppBar(context, cs),
                  const Spacer(flex: 1),
                  _buildHero(cs),
                  const Spacer(flex: 2),
                  _buildCaptureArea(cs),
                  const Spacer(flex: 1),
                  _buildFooterNav(context, cs),
                  const SizedBox(height: 24),
                ],
              ),
            ),
          ),

          // Scanning overlay
          if (_isAnalysing) _buildScanOverlay(cs),
        ],
      ),
    );
  }

  Widget _buildAppBar(BuildContext context, ColorScheme cs) {
    return Row(
      children: [
        Container(
          width: 40, height: 40,
          decoration: BoxDecoration(
            color: cs.primary,
            borderRadius: BorderRadius.circular(12),
          ),
          child: const Icon(Icons.eco_rounded, color: Colors.white, size: 22),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('CropGuard', style: Theme.of(context).textTheme.titleLarge),
              Text(
                'AI-Powered Disease Detection',
                style: Theme.of(context).textTheme.labelSmall,
              ),
            ],
          ),
        ),
        Container(
          decoration: BoxDecoration(
            color: cs.surfaceVariant,
            borderRadius: BorderRadius.circular(10),
          ),
          child: IconButton(
            icon: Icon(Icons.wifi_off_rounded, color: cs.primary, size: 20),
            onPressed: () {},
            tooltip: 'Works fully offline',
          ),
        ),
      ],
    ).animate().fadeIn(delay: 100.ms, duration: 400.ms).slideY(begin: -0.2, end: 0);
  }

  Widget _buildHero(ColorScheme cs) {
    return Center(
      child: Column(
        children: [
          // Leaf illustration container
          Container(
            width: 200,
            height: 200,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: cs.primary.withOpacity(0.06),
              border: Border.all(
                color: cs.primary.withOpacity(0.12),
                width: 1.5,
              ),
            ),
            child: Center(
              child: Stack(
                alignment: Alignment.center,
                children: [
                  Icon(
                    Icons.eco_rounded,
                    size: 100,
                    color: cs.primary.withOpacity(0.15),
                  ),
                  Icon(
                    Icons.local_florist_rounded,
                    size: 70,
                    color: cs.primary,
                  ),
                ],
              ),
            ),
          )
              .animate(onPlay: (c) => c.repeat(reverse: true))
              .scale(begin: const Offset(1, 1), end: const Offset(1.04, 1.04),
                     duration: 3000.ms, curve: Curves.easeInOut),

          const SizedBox(height: 28),
          Text(
            'Diagnose Your Crops',
            style: GoogleFonts.sora(
              fontSize: 28, fontWeight: FontWeight.w700,
              color: const Color(0xFF1A472A),
              letterSpacing: -0.5,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 10),
          Text(
            'Point your camera at any leaf to detect\n38 diseases — no internet required.',
            style: GoogleFonts.sora(
              fontSize: 14, color: const Color(0xFF5A6E50),
              height: 1.6,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    ).animate().fadeIn(delay: 200.ms, duration: 500.ms).slideY(begin: 0.1, end: 0);
  }

  Widget _buildCaptureArea(ColorScheme cs) {
    return Column(
      children: [
        // Camera button
        _ActionButton(
          label: 'Take a Photo',
          icon: Icons.camera_alt_rounded,
          primary: true,
          onPressed: () => _capture(ImageSource.camera),
        ).animate().fadeIn(delay: 350.ms, duration: 400.ms).slideY(begin: 0.2, end: 0),

        const SizedBox(height: 14),

        // Gallery button
        _ActionButton(
          label: 'Choose from Gallery',
          icon: Icons.photo_library_rounded,
          primary: false,
          onPressed: () => _capture(ImageSource.gallery),
        ).animate().fadeIn(delay: 450.ms, duration: 400.ms).slideY(begin: 0.2, end: 0),

        const SizedBox(height: 20),

        // Stats strip
        _StatsStrip(cs: cs)
            .animate().fadeIn(delay: 550.ms, duration: 400.ms),
      ],
    );
  }

  Widget _buildFooterNav(BuildContext context, ColorScheme cs) {
    return TextButton.icon(
      onPressed: () => Navigator.pushNamed(context, '/diseases'),
      icon: Icon(Icons.library_books_rounded, size: 18, color: cs.primary),
      label: Text(
        'Browse all 38 detectable diseases',
        style: GoogleFonts.sora(
          fontSize: 13, fontWeight: FontWeight.w500, color: cs.primary,
        ),
      ),
    ).animate().fadeIn(delay: 600.ms);
  }

  Widget _buildScanOverlay(ColorScheme cs) {
    return Positioned.fill(
      child: Container(
        color: Colors.black54,
        child: Center(
          child: Container(
            width: 220, height: 220,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(24),
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                SizedBox(
                  width: 64, height: 64,
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      CircularProgressIndicator(
                        strokeWidth: 3, color: cs.primary,
                      ),
                      Icon(Icons.eco_rounded, color: cs.primary, size: 28),
                    ],
                  ),
                ),
                const SizedBox(height: 20),
                AnimatedBuilder(
                  animation: _scanAnim,
                  builder: (_, __) => Text(
                    _scanAnim.value < 0.33
                        ? 'Preprocessing image…'
                        : _scanAnim.value < 0.66
                            ? 'Running AI model…'
                            : 'Generating heatmap…',
                    style: GoogleFonts.sora(
                      fontSize: 13, fontWeight: FontWeight.w500,
                      color: const Color(0xFF1A472A),
                    ),
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  'Fully offline',
                  style: GoogleFonts.sora(
                    fontSize: 11, color: const Color(0xFF6B7860),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────── sub-widgets ──────────────────────────────────── //

class _ActionButton extends StatelessWidget {
  final String  label;
  final IconData icon;
  final bool    primary;
  final VoidCallback onPressed;

  const _ActionButton({
    required this.label,
    required this.icon,
    required this.primary,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    if (primary) {
      return SizedBox(
        width: double.infinity,
        child: ElevatedButton.icon(
          onPressed: onPressed,
          icon: Icon(icon, size: 20),
          label: Text(label),
        ),
      );
    }

    return SizedBox(
      width: double.infinity,
      child: OutlinedButton.icon(
        onPressed: onPressed,
        icon: Icon(icon, size: 20),
        label: Text(label),
      ),
    );
  }
}

class _StatsStrip extends StatelessWidget {
  final ColorScheme cs;
  const _StatsStrip({required this.cs});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        _stat('38', 'diseases'),
        _divider(),
        _stat('≥90%', 'accuracy'),
        _divider(),
        _stat('100%', 'offline'),
      ],
    );
  }

  Widget _stat(String value, String label) {
    return Column(
      children: [
        Text(
          value,
          style: GoogleFonts.sora(
            fontSize: 18, fontWeight: FontWeight.w700,
            color: const Color(0xFF1A472A),
          ),
        ),
        Text(
          label,
          style: GoogleFonts.sora(
            fontSize: 11, color: const Color(0xFF6B7860),
            letterSpacing: 0.5,
          ),
        ),
      ],
    );
  }

  Widget _divider() => Padding(
    padding: const EdgeInsets.symmetric(horizontal: 20),
    child: Container(
      width: 1, height: 32,
      color: const Color(0xFFDEE5D8),
    ),
  );
}

class _BackgroundDecoration extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: -60, right: -60,
      child: Container(
        width: 240, height: 240,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: const Color(0xFF1A472A).withOpacity(0.04),
        ),
      ),
    );
  }
}
