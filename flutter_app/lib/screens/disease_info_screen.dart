// disease_info_screen.dart
// ────────────────────────────────────────────────────────────────────────────
// Screen 3 – Offline Disease Encyclopedia
//
// Lists all 38 PlantVillage classes with:
//   • Search/filter bar
//   • Plant-type chip filters
//   • Expandable treatment cards for each disease
//   • Fully offline — no network calls
// ────────────────────────────────────────────────────────────────────────────

import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';

import '../utils/classifier.dart';

// ── disease data ──────────────────────────────────────────────────────────── //

const List<String> _allLabels = [
  'Apple___Apple_scab',
  'Apple___Black_rot',
  'Apple___Cedar_apple_rust',
  'Apple___healthy',
  'Blueberry___healthy',
  'Cherry_(including_sour)___Powdery_mildew',
  'Cherry_(including_sour)___healthy',
  'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
  'Corn_(maize)___Common_rust_',
  'Corn_(maize)___Northern_Leaf_Blight',
  'Corn_(maize)___healthy',
  'Grape___Black_rot',
  'Grape___Esca_(Black_Measles)',
  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
  'Grape___healthy',
  'Orange___Haunglongbing_(Citrus_greening)',
  'Peach___Bacterial_spot',
  'Peach___healthy',
  'Pepper,_bell___Bacterial_spot',
  'Pepper,_bell___healthy',
  'Potato___Early_blight',
  'Potato___Late_blight',
  'Potato___healthy',
  'Raspberry___healthy',
  'Soybean___healthy',
  'Squash___Powdery_mildew',
  'Strawberry___Leaf_scorch',
  'Strawberry___healthy',
  'Tomato___Bacterial_spot',
  'Tomato___Early_blight',
  'Tomato___Late_blight',
  'Tomato___Leaf_Mold',
  'Tomato___Septoria_leaf_spot',
  'Tomato___Spider_mites Two-spotted_spider_mite',
  'Tomato___Target_Spot',
  'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
  'Tomato___Tomato_mosaic_virus',
  'Tomato___healthy',
];

String _plantOf(String label) => label.split('___').first
    .replaceAll('(including_sour)', '')
    .replaceAll(',_bell', ' Bell')
    .replaceAll('_(maize)', '')
    .trim();

String _diseaseOf(String label) => label.split('___').last
    .replaceAll('_', ' ')
    .split(' ')
    .map((w) => w.isEmpty ? '' : '${w[0].toUpperCase()}${w.substring(1)}')
    .join(' ');

final List<String> _plants = _allLabels
    .map(_plantOf)
    .toSet()
    .toList()
  ..sort();

const Map<String, IconData> _plantIcons = {
  'Apple':      Icons.apple,
  'Blueberry':  Icons.circle,
  'Cherry':     Icons.favorite,
  'Corn':       Icons.grass,
  'Grape':      Icons.scatter_plot,
  'Orange':     Icons.brightness_5,
  'Peach':      Icons.wb_sunny_outlined,
  'Pepper':     Icons.local_fire_department_outlined,
  'Potato':     Icons.spa_outlined,
  'Raspberry':  Icons.grain,
  'Soybean':    Icons.eco_outlined,
  'Squash':     Icons.lens,
  'Strawberry': Icons.local_florist,
  'Tomato':     Icons.circle,
};

// ── screen ───────────────────────────────────────────────────────────────── //

class DiseaseInfoScreen extends StatefulWidget {
  const DiseaseInfoScreen({super.key});

  @override
  State<DiseaseInfoScreen> createState() => _DiseaseInfoScreenState();
}

class _DiseaseInfoScreenState extends State<DiseaseInfoScreen> {
  final TextEditingController _searchCtrl = TextEditingController();
  String _query       = '';
  String? _activeFilter;               // null = all
  final Set<String> _expanded = {};

  List<String> get _filtered {
    return _allLabels.where((lbl) {
      final plant   = _plantOf(lbl);
      final disease = _diseaseOf(lbl);
      final q       = _query.toLowerCase();
      final matchQ  = q.isEmpty ||
          plant.toLowerCase().contains(q) ||
          disease.toLowerCase().contains(q);
      final matchF  = _activeFilter == null || plant == _activeFilter;
      return matchQ && matchF;
    }).toList();
  }

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      backgroundColor: cs.background,
      appBar: AppBar(
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
          'Disease Library',
          style: GoogleFonts.sora(
            fontSize: 18, fontWeight: FontWeight.w700, color: const Color(0xFF1A472A),
          ),
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Chip(
              label: Text('${_allLabels.length} classes'),
              backgroundColor: const Color(0xFF1A472A).withOpacity(0.08),
              labelStyle: GoogleFonts.sora(
                fontSize: 11, fontWeight: FontWeight.w600,
                color: const Color(0xFF1A472A),
              ),
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          _buildSearch(cs),
          _buildChips(cs),
          const Divider(height: 1),
          Expanded(
            child: _filtered.isEmpty
                ? _buildEmpty(cs)
                : ListView.builder(
                    padding: const EdgeInsets.fromLTRB(16, 8, 16, 32),
                    itemCount: _filtered.length,
                    itemBuilder: (ctx, i) =>
                        _buildDiseaseCard(ctx, _filtered[i], cs, i)
                            .animate()
                            .fadeIn(delay: (30 * i).ms, duration: 300.ms)
                            .slideX(begin: 0.04, end: 0),
                  ),
          ),
        ],
      ),
    );
  }

  // ── search bar ────────────────────────────────────────────────────────── //

  Widget _buildSearch(ColorScheme cs) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 10, 16, 6),
      child: TextField(
        controller: _searchCtrl,
        onChanged: (v) => setState(() => _query = v),
        decoration: InputDecoration(
          hintText: 'Search disease or plant…',
          hintStyle: GoogleFonts.sora(fontSize: 13, color: const Color(0xFF9AA890)),
          prefixIcon: Icon(Icons.search_rounded, color: cs.primary, size: 20),
          suffixIcon: _query.isNotEmpty
              ? IconButton(
                  icon: const Icon(Icons.clear_rounded, size: 18),
                  onPressed: () {
                    _searchCtrl.clear();
                    setState(() => _query = '');
                  },
                )
              : null,
          filled: true,
          fillColor: cs.surfaceVariant,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(14),
            borderSide: BorderSide.none,
          ),
          contentPadding: const EdgeInsets.symmetric(vertical: 12),
        ),
      ),
    );
  }

  // ── plant filter chips ────────────────────────────────────────────────── //

  Widget _buildChips(ColorScheme cs) {
    return SizedBox(
      height: 44,
      child: ListView(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 16),
        children: [
          _filterChip('All', null, cs),
          ..._plants.map((p) => _filterChip(p, p, cs)),
        ],
      ),
    );
  }

  Widget _filterChip(String label, String? value, ColorScheme cs) {
    final selected = _activeFilter == value;
    return Padding(
      padding: const EdgeInsets.only(right: 6),
      child: FilterChip(
        label: Text(label),
        selected: selected,
        onSelected: (_) => setState(() => _activeFilter = selected ? null : value),
        selectedColor: cs.primary,
        checkmarkColor: Colors.white,
        labelStyle: GoogleFonts.sora(
          fontSize: 12, fontWeight: FontWeight.w500,
          color: selected ? Colors.white : const Color(0xFF2D3A27),
        ),
        backgroundColor: cs.surfaceVariant,
        side: BorderSide.none,
        showCheckmark: false,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      ),
    );
  }

  // ── disease card ──────────────────────────────────────────────────────── //

  Widget _buildDiseaseCard(BuildContext ctx, String label, ColorScheme cs, int idx) {
    final plant    = _plantOf(label);
    final disease  = _diseaseOf(label);
    final isHealthy = label.toLowerCase().contains('healthy');
    final isOpen   = _expanded.contains(label);
    final treatment = kTreatmentDB[label] ?? 'Consult an agronomist.';

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: Column(
        children: [
          InkWell(
            borderRadius: BorderRadius.circular(20),
            onTap: () => setState(() {
              if (isOpen) {
                _expanded.remove(label);
              } else {
                _expanded.add(label);
              }
            }),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
              child: Row(
                children: [
                  // plant icon badge
                  Container(
                    width: 42, height: 42,
                    decoration: BoxDecoration(
                      color: isHealthy
                          ? const Color(0xFF2E7D32).withOpacity(0.1)
                          : const Color(0xFFD64045).withOpacity(0.08),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Icon(
                      _plantIcons[plant] ?? Icons.eco_rounded,
                      size: 22,
                      color: isHealthy
                          ? const Color(0xFF2E7D32)
                          : const Color(0xFFD64045),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          plant,
                          style: GoogleFonts.sora(
                            fontSize: 11, fontWeight: FontWeight.w600,
                            color: const Color(0xFF6B7860), letterSpacing: 0.4,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          disease,
                          style: GoogleFonts.sora(
                            fontSize: 14, fontWeight: FontWeight.w600,
                            color: const Color(0xFF1C2117),
                          ),
                        ),
                      ],
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: isHealthy
                          ? const Color(0xFF2E7D32).withOpacity(0.1)
                          : const Color(0xFFD64045).withOpacity(0.08),
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: Text(
                      isHealthy ? 'Healthy' : 'Disease',
                      style: GoogleFonts.sora(
                        fontSize: 10, fontWeight: FontWeight.w700,
                        color: isHealthy
                            ? const Color(0xFF2E7D32)
                            : const Color(0xFFD64045),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  AnimatedRotation(
                    turns: isOpen ? 0.5 : 0,
                    duration: const Duration(milliseconds: 200),
                    child: Icon(
                      Icons.keyboard_arrow_down_rounded,
                      color: cs.primary,
                      size: 22,
                    ),
                  ),
                ],
              ),
            ),
          ),
          // expanded treatment
          AnimatedCrossFade(
            firstChild: const SizedBox.shrink(),
            secondChild: Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Divider(),
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      Icon(Icons.healing_rounded,
                          size: 16, color: const Color(0xFFE8A020)),
                      const SizedBox(width: 6),
                      Text(
                        'Treatment',
                        style: GoogleFonts.sora(
                          fontSize: 12, fontWeight: FontWeight.w700,
                          color: const Color(0xFFE8A020),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    treatment,
                    style: GoogleFonts.sora(
                      fontSize: 12.5,
                      color: const Color(0xFF2D3A27),
                      height: 1.65,
                    ),
                  ),
                ],
              ),
            ),
            crossFadeState:
                isOpen ? CrossFadeState.showSecond : CrossFadeState.showFirst,
            duration: const Duration(milliseconds: 200),
          ),
        ],
      ),
    );
  }

  // ── empty state ───────────────────────────────────────────────────────── //

  Widget _buildEmpty(ColorScheme cs) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.search_off_rounded, size: 56,
              color: cs.primary.withOpacity(0.2)),
          const SizedBox(height: 12),
          Text(
            'No diseases match "$_query"',
            style: GoogleFonts.sora(fontSize: 14, color: const Color(0xFF6B7860)),
          ),
        ],
      ),
    );
  }
}
