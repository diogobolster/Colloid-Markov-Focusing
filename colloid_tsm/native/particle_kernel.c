#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

typedef struct {
    double cell_length;
    double grain_radius;
    double particle_radius;
    double mean_velocity;
    double temperature;
    double viscosity;
    double relative_permittivity;
    double ionic_strength_molar;
    double hamaker;
    double zeta_particle;
    double zeta_collector_favorable;
    double zeta_collector_unfavorable;
    double near_surface;
    double contact_gap;
    double min_gap;
    double dt;
    double max_time;
    double dlvo_velocity_cap;
    double surface_sliding_gap;
    double wall_mobility_cutoff;
    double wall_mobility_normal_floor;
    double wall_mobility_parallel_floor;
    double wall_mobility_hydro_strength;
    double diffusivity_multiplier;
    double resolved_langevin_substep_cutoff;
    double resolved_langevin_normal_step;
    double resolved_langevin_min_dt;
    double diffusivity;
    double debye_length;
    double exclusion_radius;
    double inlet_y_min;
    double inlet_y_max;
    double well_lower_gap;
    double well_upper_gap;
    double well_basin_lower_gap;
    double well_basin_upper_gap;
    double well_minimum_gap;
    double well_minimum_potential_kbt;
    double resolved_langevin_hydro_strength;
    int wall_mobility_thermal_drift;
    int resolved_langevin_substeps;
    int resolved_langevin_max_substeps;
    int has_secondary_well;
} KernelParams;

typedef struct {
    double h;
    double theta;
    double nx;
    double ny;
    double r;
    double cx;
    double cy;
    int collector;
} SurfaceState;

typedef struct {
    double x;
    double y;
    double y0;
    double t;
    int active;
    int attached;
    int exited;
    int censored;
    int interceptions;
    int in_event;
    int well_interceptions;
    int in_well_event;
    int center_interceptions;
    int corner_interceptions;
    int center_well_interceptions;
    int corner_well_interceptions;
    int contact_events;
    double near_time;
    double well_time;
    double well_angular_travel;
    double well_net_angular_travel;
    double h_min;
    double theta_entry;
    double theta_exit;
    double theta_last_near;
    double theta_final;
    double theta_well_entry;
    double theta_well_exit;
    double theta_well_last;
    double theta_well_final;
    int collector_entry;
    int collector_exit;
    int collector_last_near;
    int collector_final;
    int collector_well_entry;
    int collector_well_exit;
    int collector_well_last;
    int collector_well_final;
    double y_out;
    double travel_time;
    double dy;
    int near_wall_outer_steps;
    int adaptive_substeps;
    int guard_hits;
    int min_dt_hits;
    double sub_dt_sum;
    double sub_dt_sq_sum;
    double min_sub_dt;
    double max_sub_dt;
    double max_inward_det_normal_step;
    double max_brownian_normal_std;
    double first_entry_theta;
    double first_entry_time;
    double first_entry_x;
    double first_entry_y;
    double first_entry_collector_cx;
    double first_entry_collector_cy;
    int first_entry_collector;
    double next_entry_theta;
    double next_entry_time;
    double next_entry_x;
    double next_entry_y;
    double next_entry_collector_cx;
    double next_entry_collector_cy;
    int next_entry_collector;
    double departure_theta_to_next;
    double departure_time_to_next;
    double first_collector_near_time;
    int same_collector_reentries_before_next;
} ParticleState;

typedef struct {
    double sub_dt;
    double normal_mobility;
    double normal_speed;
    double target;
    int min_dt_limited;
} NearWallStepChoice;

static double positive_mod(double value, double length) {
    double out = fmod(value, length);
    if (out < 0.0) {
        out += length;
    }
    return out;
}

static int wrap_index(int index, int n) {
    int out = index % n;
    if (out < 0) {
        out += n;
    }
    return out;
}

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static double rng_uniform(uint64_t *state) {
    uint64_t z = splitmix64(state);
    double u = (double)(z >> 11) * (1.0 / 9007199254740992.0);
    if (u <= 0.0) {
        return 0x1p-53;
    }
    if (u >= 1.0) {
        return 1.0 - 0x1p-53;
    }
    return u;
}

static void rng_normal_pair(uint64_t *state, double *z0, double *z1) {
    double u1 = rng_uniform(state);
    double u2 = rng_uniform(state);
    double radius = sqrt(-2.0 * log(u1));
    double angle = 2.0 * M_PI * u2;
    *z0 = radius * cos(angle);
    *z1 = radius * sin(angle);
}

static void nearest_grain_state(const KernelParams *p, double x, double y, SurfaceState *s) {
    double length = p->cell_length;

    double corner_cx = length * round(x / length);
    double corner_cy = length * round(y / length);
    double corner_rx = x - corner_cx;
    double corner_ry = y - corner_cy;
    double corner_r2 = corner_rx * corner_rx + corner_ry * corner_ry;

    double center_cx = length * (round((x - 0.5 * length) / length) + 0.5);
    double center_cy = length * (round((y - 0.5 * length) / length) + 0.5);
    double center_rx = x - center_cx;
    double center_ry = y - center_cy;
    double center_r2 = center_rx * center_rx + center_ry * center_ry;

    int use_center = center_r2 <= corner_r2;
    double rx = use_center ? center_rx : corner_rx;
    double ry = use_center ? center_ry : corner_ry;
    double r2 = use_center ? center_r2 : corner_r2;
    double r = sqrt(r2);
    if (r < 1.0e-30) {
        r = 1.0e-30;
    }

    s->h = r - p->exclusion_radius;
    s->theta = positive_mod(atan2(ry, rx), 2.0 * M_PI);
    s->nx = rx / r;
    s->ny = ry / r;
    s->r = r;
    s->cx = use_center ? center_cx : corner_cx;
    s->cy = use_center ? center_cy : corner_cy;
    s->collector = use_center ? 1 : 0;
}

static void project_to_exclusion_surface(const KernelParams *p, double *x, double *y) {
    SurfaceState s;
    nearest_grain_state(p, *x, *y, &s);
    double target = p->exclusion_radius + p->contact_gap;
    *x = s.cx + s.nx * target;
    *y = s.cy + s.ny * target;
}

static int in_secondary_well(const KernelParams *p, double h) {
    if (!p->has_secondary_well) {
        return 0;
    }
    return h >= p->well_lower_gap && h <= p->well_upper_gap;
}

static int in_secondary_basin(const KernelParams *p, double h) {
    if (!p->has_secondary_well) {
        return 0;
    }
    return h >= p->well_basin_lower_gap && h <= p->well_basin_upper_gap;
}

static double angular_delta(double theta, double previous_theta) {
    return atan2(sin(theta - previous_theta), cos(theta - previous_theta));
}

static int record_entry_pair(
    ParticleState *particle,
    double theta,
    int collector,
    double x,
    double y,
    double time,
    double collector_cx,
    double collector_cy,
    int periodic_image_mode
) {
    if (particle->first_entry_collector < 0) {
        particle->first_entry_theta = theta;
        particle->first_entry_time = time;
        particle->first_entry_x = x;
        particle->first_entry_y = y;
        particle->first_entry_collector_cx = collector_cx;
        particle->first_entry_collector_cy = collector_cy;
        particle->first_entry_collector = collector;
        return 0;
    }
    if (particle->next_entry_collector >= 0) {
        return 0;
    }
    int same_collector = collector == particle->first_entry_collector;
    if (same_collector && periodic_image_mode > 0) {
        double tolerance = 1.0e-9;
        same_collector = fabs(collector_cx - particle->first_entry_collector_cx) <= tolerance;
        if (same_collector && periodic_image_mode == 1) {
            same_collector = fabs(collector_cy - particle->first_entry_collector_cy) <= tolerance;
        }
    }
    if (same_collector) {
        particle->same_collector_reentries_before_next += 1;
        return 0;
    }
    particle->next_entry_theta = theta;
    particle->next_entry_time = time;
    particle->next_entry_x = x;
    particle->next_entry_y = y;
    particle->next_entry_collector_cx = collector_cx;
    particle->next_entry_collector_cy = collector_cy;
    particle->next_entry_collector = collector;
    return 1;
}

static void periodic_shell_entry_state(
    const KernelParams *p,
    double old_x,
    double old_y,
    double new_x,
    double new_y,
    const SurfaceState *s0,
    const SurfaceState *s1,
    SurfaceState *entry,
    double *fraction
) {
    if (s0->h <= p->near_surface) {
        *entry = *s0;
        *fraction = 0.0;
        return;
    }
    if (s1->h > p->near_surface) {
        *entry = *s1;
        *fraction = 1.0;
        return;
    }
    double lo = 0.0;
    double hi = 1.0;
    SurfaceState candidate = *s1;
    for (int iteration = 0; iteration < 28; ++iteration) {
        double mid = 0.5 * (lo + hi);
        nearest_grain_state(
            p,
            old_x + mid * (new_x - old_x),
            old_y + mid * (new_y - old_y),
            &candidate
        );
        if (candidate.h > p->near_surface) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    *fraction = hi;
    nearest_grain_state(
        p,
        old_x + hi * (new_x - old_x),
        old_y + hi * (new_y - old_y),
        entry
    );
}

static void interpolate_velocity(
    const KernelParams *p,
    int n,
    const double *ux,
    const double *uy,
    double x,
    double y,
    double *out_ux,
    double *out_uy
) {
    double gx = positive_mod(x, p->cell_length) / p->cell_length * (double)n - 0.5;
    double gy = positive_mod(y, p->cell_length) / p->cell_length * (double)n - 0.5;
    int i0_raw = (int)floor(gx);
    int j0_raw = (int)floor(gy);
    double tx = gx - (double)i0_raw;
    double ty = gy - (double)j0_raw;
    int i0 = wrap_index(i0_raw, n);
    int j0 = wrap_index(j0_raw, n);
    int i1 = wrap_index(i0 + 1, n);
    int j1 = wrap_index(j0 + 1, n);

    size_t i00 = (size_t)i0 * (size_t)n + (size_t)j0;
    size_t i10 = (size_t)i1 * (size_t)n + (size_t)j0;
    size_t i01 = (size_t)i0 * (size_t)n + (size_t)j1;
    size_t i11 = (size_t)i1 * (size_t)n + (size_t)j1;
    double w00 = (1.0 - tx) * (1.0 - ty);
    double w10 = tx * (1.0 - ty);
    double w01 = (1.0 - tx) * ty;
    double w11 = tx * ty;

    *out_ux = w00 * ux[i00] + w10 * ux[i10] + w01 * ux[i01] + w11 * ux[i11];
    *out_uy = w00 * uy[i00] + w10 * uy[i10] + w01 * uy[i01] + w11 * uy[i11];
}

static void near_wall_mobility(
    const KernelParams *p,
    double h,
    double *normal,
    double *parallel,
    double *dnormal_dh
) {
    double h_eff = fmax(h, p->min_gap);
    int active = h <= p->wall_mobility_cutoff;
    double a = p->particle_radius;

    double normal_raw = h_eff / (h_eff + a);
    double normal_value = 1.0;
    double dnormal_value = 0.0;
    if (active) {
        normal_value = fmin(fmax(normal_raw, p->wall_mobility_normal_floor), 1.0);
        if (normal_raw > p->wall_mobility_normal_floor) {
            dnormal_value = a / ((h_eff + a) * (h_eff + a));
        }
    }

    double alpha = a / (a + h_eff);
    if (alpha < 0.0) {
        alpha = 0.0;
    }
    if (alpha > 0.999999) {
        alpha = 0.999999;
    }
    double alpha2 = alpha * alpha;
    double alpha3 = alpha2 * alpha;
    double alpha4 = alpha3 * alpha;
    double alpha5 = alpha4 * alpha;
    double parallel_far = (
        1.0
        - 9.0 / 16.0 * alpha
        + 1.0 / 8.0 * alpha3
        - 45.0 / 256.0 * alpha4
        - 1.0 / 16.0 * alpha5
    );
    double log_ratio = log(a / h_eff);
    if (log_ratio < 0.0) {
        log_ratio = 0.0;
    }
    double parallel_near = 1.0 / (1.0 + 0.9588 * log_ratio);
    double parallel_raw = fmin(parallel_far, parallel_near);
    double parallel_value = 1.0;
    if (active) {
        parallel_value = fmin(fmax(parallel_raw, p->wall_mobility_parallel_floor), 1.0);
    }

    *normal = normal_value;
    *parallel = parallel_value;
    *dnormal_dh = dnormal_value;
}

static double dlvo_force_normal(const KernelParams *p, double h, int condition) {
    if (condition == 0) {
        return 0.0;
    }
    if (h > p->near_surface) {
        return 0.0;
    }
    double h_eff = fmax(h, p->min_gap);
    double eps0 = 8.8541878128e-12;
    double eps = p->relative_permittivity * eps0;
    double kappa = 1.0 / p->debye_length;
    double zeta_c = condition == 1 ? p->zeta_collector_favorable : p->zeta_collector_unfavorable;
    double f_edl = (
        2.0
        * M_PI
        * eps
        * p->particle_radius
        * kappa
        * p->zeta_particle
        * zeta_c
        * exp(-kappa * h_eff)
    );
    double f_vdw = -p->hamaker * p->particle_radius / (6.0 * h_eff * h_eff);
    return f_edl + f_vdw;
}

static double dlvo_potential_energy(const KernelParams *p, double h, int condition) {
    if (condition == 0) {
        return 0.0;
    }
    double h_eff = fmax(h, p->min_gap);
    double eps0 = 8.8541878128e-12;
    double eps = p->relative_permittivity * eps0;
    double kappa = 1.0 / p->debye_length;
    double zeta_c = condition == 1 ? p->zeta_collector_favorable : p->zeta_collector_unfavorable;
    double c_edl = (
        2.0
        * M_PI
        * eps
        * p->particle_radius
        * kappa
        * p->zeta_particle
        * zeta_c
    );
    double u_edl = c_edl / kappa * exp(-kappa * h_eff);
    double u_vdw = -p->hamaker * p->particle_radius / (6.0 * h_eff);
    return u_edl + u_vdw;
}

static NearWallStepChoice resolved_near_wall_choice(
    const KernelParams *p,
    int ngrid,
    const double *ux_grid,
    const double *uy_grid,
    int condition,
    const ParticleState *particle,
    double remaining_dt
) {
    NearWallStepChoice choice;
    choice.sub_dt = remaining_dt;
    choice.normal_mobility = 1.0;
    choice.normal_speed = 0.0;
    choice.target = fmax(p->resolved_langevin_normal_step, p->min_gap * 0.05);
    choice.min_dt_limited = 0;
    if (!particle->active) {
        return choice;
    }
    SurfaceState s;
    nearest_grain_state(p, particle->x, particle->y, &s);
    double ux = 0.0;
    double uy = 0.0;
    interpolate_velocity(p, ngrid, ux_grid, uy_grid, particle->x, particle->y, &ux, &uy);
    double normal_mobility = 1.0;
    double parallel_mobility = 1.0;
    double dnormal_dh = 0.0;
    near_wall_mobility(p, s.h, &normal_mobility, &parallel_mobility, &dnormal_dh);
    (void)parallel_mobility;

    double hydro_normal = ux * s.nx + uy * s.ny;
    double force_normal = dlvo_force_normal(p, s.h, condition);
    double stokes_mobility = 1.0 / (6.0 * M_PI * p->viscosity * p->particle_radius);
    double hydro_strength = p->resolved_langevin_hydro_strength;
    double hydro_normal_scale = 1.0 - hydro_strength * (1.0 - normal_mobility);
    double normal_speed = hydro_normal * hydro_normal_scale + stokes_mobility * force_normal * normal_mobility;
    if (p->wall_mobility_thermal_drift) {
        normal_speed += p->diffusivity * dnormal_dh;
    }

    double target = fmax(p->resolved_langevin_normal_step, p->min_gap * 0.05);
    double inward_speed = fmax(-normal_speed, 0.0);
    double dt_det = remaining_dt;
    if (inward_speed > 1.0e-30) {
        dt_det = 0.25 * target / inward_speed;
    }
    double dt_diff = remaining_dt;
    if (p->diffusivity > 0.0 && normal_mobility > 0.0) {
        dt_diff = 0.25 * target * target / (2.0 * p->diffusivity * normal_mobility);
    }
    int substeps = p->resolved_langevin_substeps > 0 ? p->resolved_langevin_substeps : 1;
    double base_dt = p->dt / (double)substeps;
    double out = remaining_dt;
    if (base_dt < out) {
        out = base_dt;
    }
    if (dt_det < out) {
        out = dt_det;
    }
    if (dt_diff < out) {
        out = dt_diff;
    }
    if (out < p->resolved_langevin_min_dt) {
        out = p->resolved_langevin_min_dt;
        choice.min_dt_limited = 1;
    }
    if (out > remaining_dt) {
        out = remaining_dt;
    }
    choice.sub_dt = out;
    choice.normal_mobility = normal_mobility;
    choice.normal_speed = normal_speed;
    choice.target = target;
    return choice;
}

static double resolved_near_wall_dt(
    const KernelParams *p,
    int ngrid,
    const double *ux_grid,
    const double *uy_grid,
    int condition,
    const ParticleState *particle,
    double remaining_dt
) {
    NearWallStepChoice choice = resolved_near_wall_choice(
        p,
        ngrid,
        ux_grid,
        uy_grid,
        condition,
        particle,
        remaining_dt
    );
    return choice.sub_dt;
}

static void advance_step(
    const KernelParams *p,
    int ngrid,
    const double *ux_grid,
    const double *uy_grid,
    int condition,
    int surface_mode,
    int allow_attachment,
    int stop_at_next_distinct,
    double step_dt,
    uint64_t *rng_state,
    ParticleState *particle
) {
    if (!particle->active || step_dt <= 0.0) {
        return;
    }

    double old_x = particle->x;
    double old_y = particle->y;
    SurfaceState s0;
    nearest_grain_state(p, old_x, old_y, &s0);

    double ux = 0.0;
    double uy = 0.0;
    interpolate_velocity(p, ngrid, ux_grid, uy_grid, old_x, old_y, &ux, &uy);

    double vx = 0.0;
    double vy = 0.0;
    double force_normal = dlvo_force_normal(p, s0.h, condition);
    double stokes_mobility = 1.0 / (6.0 * M_PI * p->viscosity * p->particle_radius);
    if (surface_mode == 3) {
        double speed = stokes_mobility * force_normal;
        vx = speed * s0.nx;
        vy = speed * s0.ny;
    } else {
        double speed = stokes_mobility * force_normal;
        if (speed > p->dlvo_velocity_cap) {
            speed = p->dlvo_velocity_cap;
        }
        if (speed < -p->dlvo_velocity_cap) {
            speed = -p->dlvo_velocity_cap;
        }
        vx = speed * s0.nx;
        vy = speed * s0.ny;
    }

    int use_wall_langevin = surface_mode == 2 || surface_mode == 3;
    double normal_mobility = 1.0;
    double parallel_mobility = 1.0;
    double dnormal_dh = 0.0;
    double tx = -s0.ny;
    double ty = s0.nx;
    if (use_wall_langevin) {
        near_wall_mobility(p, s0.h, &normal_mobility, &parallel_mobility, &dnormal_dh);

        double hydro_normal = ux * s0.nx + uy * s0.ny;
        double hydro_parallel = ux * tx + uy * ty;
        double hydro_strength = surface_mode == 3 ? p->resolved_langevin_hydro_strength : p->wall_mobility_hydro_strength;
        double hydro_normal_scale = 1.0 - hydro_strength * (1.0 - normal_mobility);
        double hydro_parallel_scale = 1.0 - hydro_strength * (1.0 - parallel_mobility);
        ux = hydro_normal * hydro_normal_scale * s0.nx + hydro_parallel * hydro_parallel_scale * tx;
        uy = hydro_normal * hydro_normal_scale * s0.ny + hydro_parallel * hydro_parallel_scale * ty;

        double dlvo_normal = vx * s0.nx + vy * s0.ny;
        double dlvo_parallel = vx * tx + vy * ty;
        vx = dlvo_normal * normal_mobility * s0.nx + dlvo_parallel * parallel_mobility * tx;
        vy = dlvo_normal * normal_mobility * s0.ny + dlvo_parallel * parallel_mobility * ty;
        if (p->wall_mobility_thermal_drift) {
            double thermal_drift = p->diffusivity * dnormal_dh;
            vx += thermal_drift * s0.nx;
            vy += thermal_drift * s0.ny;
        }
    }

    double vtot_x = ux + vx;
    double vtot_y = uy + vy;
    if (surface_mode == 1 && !allow_attachment && condition == 1 && s0.h <= p->contact_gap + p->surface_sliding_gap) {
        double normal_velocity = vtot_x * s0.nx + vtot_y * s0.ny;
        if (normal_velocity < 0.0) {
            vtot_x -= normal_velocity * s0.nx;
            vtot_y -= normal_velocity * s0.ny;
        }
    }

    double noise0 = 0.0;
    double noise1 = 0.0;
    rng_normal_pair(rng_state, &noise0, &noise1);

    double new_x = old_x;
    double new_y = old_y;
    if (use_wall_langevin) {
        double brownian_normal = sqrt(2.0 * p->diffusivity * normal_mobility * step_dt) * noise0;
        double brownian_parallel = sqrt(2.0 * p->diffusivity * parallel_mobility * step_dt) * noise1;
        new_x += vtot_x * step_dt + brownian_normal * s0.nx + brownian_parallel * tx;
        new_y += vtot_y * step_dt + brownian_normal * s0.ny + brownian_parallel * ty;
    } else {
        double sqrt_2d_step = sqrt(2.0 * p->diffusivity * step_dt);
        new_x += vtot_x * step_dt + sqrt_2d_step * noise0;
        new_y += vtot_y * step_dt + sqrt_2d_step * noise1;
    }

    SurfaceState s1;
    nearest_grain_state(p, new_x, new_y, &s1);
    if (surface_mode == 3 && condition == 2 && s1.h < s0.h) {
        double old_u = dlvo_potential_energy(p, s0.h, condition);
        double new_u = dlvo_potential_energy(p, s1.h, condition);
        if (new_u > old_u) {
            double kbt = 1.380649e-23 * p->temperature;
            double exponent = (new_u - old_u) / kbt;
            if (exponent > 700.0) {
                exponent = 700.0;
            }
            double accept_probability = exp(-exponent);
            if (rng_uniform(rng_state) > accept_probability) {
                new_x = old_x;
                new_y = old_y;
                nearest_grain_state(p, new_x, new_y, &s1);
            }
        }
    }

    int inside = s1.h < p->contact_gap;
    int hit = 0;
    if (inside) {
        particle->contact_events += 1;
        if (allow_attachment) {
            hit = 1;
            particle->attached = 1;
            particle->active = 0;
        }
    }
    if (inside && !hit) {
        project_to_exclusion_surface(p, &new_x, &new_y);
        nearest_grain_state(p, new_x, new_y, &s1);
    }

    particle->x = new_x;
    particle->y = new_y;
    particle->t += step_dt;

    if (s1.h < particle->h_min) {
        particle->h_min = s1.h;
    }
    int now_near = s1.h <= p->near_surface;
    int entering = now_near && !particle->in_event;
    int leaving = !now_near && particle->in_event;
    if (leaving) {
        particle->theta_exit = particle->theta_last_near;
        particle->collector_exit = particle->collector_last_near;
        if (
            particle->next_entry_collector < 0 &&
            particle->collector_last_near == particle->first_entry_collector
        ) {
            particle->departure_theta_to_next = particle->theta_last_near;
            particle->departure_time_to_next = particle->t;
        }
    }
    int completed_entry_pair = 0;
    if (entering) {
        SurfaceState entry_state;
        double entry_fraction = 1.0;
        periodic_shell_entry_state(
            p,
            old_x,
            old_y,
            new_x,
            new_y,
            &s0,
            &s1,
            &entry_state,
            &entry_fraction
        );
        double entry_x = old_x + entry_fraction * (new_x - old_x);
        double entry_y = old_y + entry_fraction * (new_y - old_y);
        double entry_time = particle->t - step_dt + entry_fraction * step_dt;
        particle->interceptions += 1;
        particle->theta_entry = entry_state.theta;
        particle->collector_entry = entry_state.collector;
        completed_entry_pair = record_entry_pair(
            particle,
            entry_state.theta,
            entry_state.collector,
            entry_x,
            entry_y,
            entry_time,
            entry_state.cx,
            entry_state.cy,
            1
        );
        if (entry_state.collector == 1) {
            particle->center_interceptions += 1;
        } else {
            particle->corner_interceptions += 1;
        }
    }
    if (now_near) {
        particle->theta_last_near = s1.theta;
        particle->collector_last_near = s1.collector;
        particle->near_time += step_dt;
        if (
            particle->next_entry_collector < 0 &&
            s1.collector == particle->first_entry_collector
        ) {
            particle->first_collector_near_time += step_dt;
        }
    }
    particle->in_event = now_near;

    if (completed_entry_pair && stop_at_next_distinct) {
        particle->active = 0;
        particle->travel_time = particle->t;
        particle->dy = particle->y - particle->y0;
    }

    int in_entry_shell = in_secondary_well(p, s1.h);
    int in_basin = in_secondary_basin(p, s1.h);
    int was_well = particle->in_well_event;
    int now_well = was_well ? in_basin : in_entry_shell;
    int well_entering = in_entry_shell && !was_well;
    int well_leaving = !in_basin && was_well;
    if (was_well && !isnan(particle->theta_well_last)) {
        double dtheta = angular_delta(s1.theta, particle->theta_well_last);
        particle->well_angular_travel += fabs(dtheta);
        particle->well_net_angular_travel += dtheta;
    }
    if (well_leaving) {
        particle->theta_well_exit = s1.theta;
        particle->collector_well_exit = s1.collector;
    }
    if (well_entering) {
        particle->well_interceptions += 1;
        particle->theta_well_entry = s1.theta;
        particle->collector_well_entry = s1.collector;
        if (s1.collector == 1) {
            particle->center_well_interceptions += 1;
        } else {
            particle->corner_well_interceptions += 1;
        }
    }
    if (now_well) {
        particle->theta_well_last = s1.theta;
        particle->collector_well_last = s1.collector;
        particle->well_time += step_dt;
    }
    particle->in_well_event = now_well;

    if (particle->active && !stop_at_next_distinct && particle->x >= p->cell_length) {
        particle->exited = 1;
        particle->active = 0;
        particle->y_out = positive_mod(particle->y, p->cell_length);
        particle->travel_time = particle->t;
        particle->dy = particle->y - particle->y0;
    }
}

int simulate_particle_kernel(
    const KernelParams *p,
    int ngrid,
    const double *ux_grid,
    const double *uy_grid,
    int condition,
    int surface_mode,
    int allow_attachment,
    int stop_at_next_distinct,
    int n_particles,
    uint64_t seed,
    const double *initial_x,
    const double *initial_y,
    double *y_in,
    double *y_out,
    double *travel_time,
    double *dy,
    uint8_t *attached,
    uint8_t *exited,
    uint8_t *censored,
    int32_t *interceptions,
    double *near_time,
    double *h_min,
    double *theta_entry,
    double *theta_exit,
    double *theta_final,
    int32_t *collector_entry,
    int32_t *collector_exit,
    int32_t *collector_final,
    int32_t *center_interceptions,
    int32_t *corner_interceptions,
    int32_t *contact_events,
    double *x_final,
    double *y_final,
    int32_t *well_interceptions,
    double *well_time,
    double *theta_well_entry,
    double *theta_well_exit,
    double *theta_well_final,
    int32_t *collector_well_entry,
    int32_t *collector_well_exit,
    int32_t *collector_well_final,
    int32_t *center_well_interceptions,
    int32_t *corner_well_interceptions,
    double *well_angular_travel,
    double *well_net_angular_travel,
    int32_t *near_wall_outer_steps,
    int32_t *adaptive_substeps,
    int32_t *guard_hits,
    int32_t *min_dt_hits,
    double *sub_dt_sum,
    double *sub_dt_sq_sum,
    double *min_sub_dt,
    double *max_sub_dt,
    double *max_inward_det_normal_step,
    double *max_brownian_normal_std,
    double *first_entry_theta,
    double *first_entry_time,
    double *first_entry_x,
    double *first_entry_y,
    double *first_entry_collector_cx,
    double *first_entry_collector_cy,
    int32_t *first_entry_collector,
    double *next_entry_theta,
    double *next_entry_time,
    double *next_entry_x,
    double *next_entry_y,
    double *next_entry_collector_cx,
    double *next_entry_collector_cy,
    int32_t *next_entry_collector,
    double *departure_theta_to_next,
    double *departure_time_to_next,
    double *first_collector_near_time,
    int32_t *same_collector_reentries_before_next
) {
    if (p == 0 || ux_grid == 0 || uy_grid == 0 || initial_x == 0 || initial_y == 0) {
        return 1;
    }
    if (ngrid <= 1 || n_particles < 0 || p->dt <= 0.0 || p->max_time <= 0.0) {
        return 2;
    }
    if (!(surface_mode == 1 || surface_mode == 2 || surface_mode == 3)) {
        return 3;
    }

    int max_steps = (int)ceil(p->max_time / p->dt);
    if (max_steps < 1) {
        max_steps = 1;
    }

    for (int particle_id = 0; particle_id < n_particles; ++particle_id) {
        ParticleState particle;
        particle.x = initial_x[particle_id];
        particle.y = initial_y[particle_id];
        particle.y0 = initial_y[particle_id];
        particle.t = 0.0;
        particle.active = 1;
        particle.attached = 0;
        particle.exited = 0;
        particle.censored = 0;
        particle.interceptions = 0;
        particle.in_event = 0;
        particle.well_interceptions = 0;
        particle.in_well_event = 0;
        particle.center_interceptions = 0;
        particle.corner_interceptions = 0;
        particle.center_well_interceptions = 0;
        particle.corner_well_interceptions = 0;
        particle.contact_events = 0;
        particle.near_time = 0.0;
        particle.well_time = 0.0;
        particle.well_angular_travel = 0.0;
        particle.well_net_angular_travel = 0.0;
        particle.h_min = INFINITY;
        particle.theta_entry = NAN;
        particle.theta_exit = NAN;
        particle.theta_last_near = NAN;
        particle.theta_final = NAN;
        particle.theta_well_entry = NAN;
        particle.theta_well_exit = NAN;
        particle.theta_well_last = NAN;
        particle.theta_well_final = NAN;
        particle.collector_entry = -1;
        particle.collector_exit = -1;
        particle.collector_last_near = -1;
        particle.collector_final = -1;
        particle.collector_well_entry = -1;
        particle.collector_well_exit = -1;
        particle.collector_well_last = -1;
        particle.collector_well_final = -1;
        particle.y_out = NAN;
        particle.travel_time = NAN;
        particle.dy = NAN;
        particle.near_wall_outer_steps = 0;
        particle.adaptive_substeps = 0;
        particle.guard_hits = 0;
        particle.min_dt_hits = 0;
        particle.sub_dt_sum = 0.0;
        particle.sub_dt_sq_sum = 0.0;
        particle.min_sub_dt = INFINITY;
        particle.max_sub_dt = 0.0;
        particle.max_inward_det_normal_step = 0.0;
        particle.max_brownian_normal_std = 0.0;
        particle.first_entry_theta = NAN;
        particle.first_entry_time = NAN;
        particle.first_entry_x = NAN;
        particle.first_entry_y = NAN;
        particle.first_entry_collector_cx = NAN;
        particle.first_entry_collector_cy = NAN;
        particle.first_entry_collector = -1;
        particle.next_entry_theta = NAN;
        particle.next_entry_time = NAN;
        particle.next_entry_x = NAN;
        particle.next_entry_y = NAN;
        particle.next_entry_collector_cx = NAN;
        particle.next_entry_collector_cy = NAN;
        particle.next_entry_collector = -1;
        particle.departure_theta_to_next = NAN;
        particle.departure_time_to_next = NAN;
        particle.first_collector_near_time = 0.0;
        particle.same_collector_reentries_before_next = 0;

        uint64_t rng_state = seed ^ (UINT64_C(0xD1B54A32D192ED03) * (uint64_t)(particle_id + 1));
        rng_state += UINT64_C(0x9E3779B97F4A7C15);

        for (int step = 0; step < max_steps && particle.active; ++step) {
            SurfaceState current;
            nearest_grain_state(p, particle.x, particle.y, &current);
            if (surface_mode == 3 && current.h <= p->resolved_langevin_substep_cutoff) {
                particle.near_wall_outer_steps += 1;
                double remaining_dt = p->dt;
                int guard_max = p->resolved_langevin_max_substeps > 0 ? p->resolved_langevin_max_substeps : 1;
                int guard = 0;
                while (remaining_dt > 1.0e-15 && particle.active && guard < guard_max) {
                    NearWallStepChoice step_choice = resolved_near_wall_choice(
                        p,
                        ngrid,
                        ux_grid,
                        uy_grid,
                        condition,
                        &particle,
                        remaining_dt
                    );
                    double sub_dt = step_choice.sub_dt;
                    if (sub_dt > remaining_dt) {
                        sub_dt = remaining_dt;
                    }
                    if (sub_dt < particle.min_sub_dt) {
                        particle.min_sub_dt = sub_dt;
                    }
                    if (sub_dt > particle.max_sub_dt) {
                        particle.max_sub_dt = sub_dt;
                    }
                    particle.sub_dt_sum += sub_dt;
                    particle.sub_dt_sq_sum += sub_dt * sub_dt;
                    particle.adaptive_substeps += 1;
                    particle.min_dt_hits += step_choice.min_dt_limited;
                    double det_normal_step = fmax(-step_choice.normal_speed, 0.0) * sub_dt;
                    if (det_normal_step > particle.max_inward_det_normal_step) {
                        particle.max_inward_det_normal_step = det_normal_step;
                    }
                    double brownian_std = 0.0;
                    if (p->diffusivity > 0.0 && step_choice.normal_mobility > 0.0) {
                        brownian_std = sqrt(2.0 * p->diffusivity * step_choice.normal_mobility * sub_dt);
                    }
                    if (brownian_std > particle.max_brownian_normal_std) {
                        particle.max_brownian_normal_std = brownian_std;
                    }
                    advance_step(
                        p,
                        ngrid,
                        ux_grid,
                        uy_grid,
                        condition,
                        surface_mode,
                        allow_attachment,
                        stop_at_next_distinct,
                        sub_dt,
                        &rng_state,
                        &particle
                    );
                    remaining_dt -= sub_dt;
                    ++guard;
                }
                if (remaining_dt > 1.0e-15 && particle.active) {
                    particle.guard_hits += 1;
                    advance_step(
                        p,
                        ngrid,
                        ux_grid,
                        uy_grid,
                        condition,
                        surface_mode,
                        allow_attachment,
                        stop_at_next_distinct,
                        remaining_dt,
                        &rng_state,
                        &particle
                    );
                }
            } else {
                advance_step(
                    p,
                    ngrid,
                    ux_grid,
                    uy_grid,
                    condition,
                    surface_mode,
                    allow_attachment,
                    stop_at_next_distinct,
                    p->dt,
                    &rng_state,
                    &particle
                );
            }
        }

        if (particle.active) {
            particle.censored = 1;
        }
        SurfaceState final_state;
        nearest_grain_state(p, particle.x, particle.y, &final_state);
        particle.theta_final = final_state.theta;
        particle.collector_final = final_state.collector;
        particle.theta_well_final = final_state.theta;
        particle.collector_well_final = final_state.collector;

        y_in[particle_id] = particle.y0;
        y_out[particle_id] = particle.y_out;
        travel_time[particle_id] = particle.travel_time;
        dy[particle_id] = particle.dy;
        attached[particle_id] = (uint8_t)particle.attached;
        exited[particle_id] = (uint8_t)particle.exited;
        censored[particle_id] = (uint8_t)particle.censored;
        interceptions[particle_id] = (int32_t)particle.interceptions;
        near_time[particle_id] = particle.near_time;
        h_min[particle_id] = particle.h_min;
        theta_entry[particle_id] = particle.theta_entry;
        theta_exit[particle_id] = particle.theta_exit;
        theta_final[particle_id] = particle.theta_final;
        collector_entry[particle_id] = (int32_t)particle.collector_entry;
        collector_exit[particle_id] = (int32_t)particle.collector_exit;
        collector_final[particle_id] = (int32_t)particle.collector_final;
        center_interceptions[particle_id] = (int32_t)particle.center_interceptions;
        corner_interceptions[particle_id] = (int32_t)particle.corner_interceptions;
        contact_events[particle_id] = (int32_t)particle.contact_events;
        x_final[particle_id] = particle.x;
        y_final[particle_id] = particle.y;
        well_interceptions[particle_id] = (int32_t)particle.well_interceptions;
        well_time[particle_id] = particle.well_time;
        theta_well_entry[particle_id] = particle.theta_well_entry;
        theta_well_exit[particle_id] = particle.theta_well_exit;
        theta_well_final[particle_id] = particle.theta_well_final;
        collector_well_entry[particle_id] = (int32_t)particle.collector_well_entry;
        collector_well_exit[particle_id] = (int32_t)particle.collector_well_exit;
        collector_well_final[particle_id] = (int32_t)particle.collector_well_final;
        center_well_interceptions[particle_id] = (int32_t)particle.center_well_interceptions;
        corner_well_interceptions[particle_id] = (int32_t)particle.corner_well_interceptions;
        well_angular_travel[particle_id] = particle.well_angular_travel;
        well_net_angular_travel[particle_id] = particle.well_net_angular_travel;
        near_wall_outer_steps[particle_id] = (int32_t)particle.near_wall_outer_steps;
        adaptive_substeps[particle_id] = (int32_t)particle.adaptive_substeps;
        guard_hits[particle_id] = (int32_t)particle.guard_hits;
        min_dt_hits[particle_id] = (int32_t)particle.min_dt_hits;
        sub_dt_sum[particle_id] = particle.sub_dt_sum;
        sub_dt_sq_sum[particle_id] = particle.sub_dt_sq_sum;
        min_sub_dt[particle_id] = isfinite(particle.min_sub_dt) ? particle.min_sub_dt : NAN;
        max_sub_dt[particle_id] = particle.max_sub_dt;
        max_inward_det_normal_step[particle_id] = particle.max_inward_det_normal_step;
        max_brownian_normal_std[particle_id] = particle.max_brownian_normal_std;
        first_entry_theta[particle_id] = particle.first_entry_theta;
        first_entry_time[particle_id] = particle.first_entry_time;
        first_entry_x[particle_id] = particle.first_entry_x;
        first_entry_y[particle_id] = particle.first_entry_y;
        first_entry_collector_cx[particle_id] = particle.first_entry_collector_cx;
        first_entry_collector_cy[particle_id] = particle.first_entry_collector_cy;
        first_entry_collector[particle_id] = (int32_t)particle.first_entry_collector;
        next_entry_theta[particle_id] = particle.next_entry_theta;
        next_entry_time[particle_id] = particle.next_entry_time;
        next_entry_x[particle_id] = particle.next_entry_x;
        next_entry_y[particle_id] = particle.next_entry_y;
        next_entry_collector_cx[particle_id] = particle.next_entry_collector_cx;
        next_entry_collector_cy[particle_id] = particle.next_entry_collector_cy;
        next_entry_collector[particle_id] = (int32_t)particle.next_entry_collector;
        departure_theta_to_next[particle_id] = particle.departure_theta_to_next;
        departure_time_to_next[particle_id] = particle.departure_time_to_next;
        first_collector_near_time[particle_id] = particle.first_collector_near_time;
        same_collector_reentries_before_next[particle_id] =
            (int32_t)particle.same_collector_reentries_before_next;
    }

    return 0;
}

typedef struct {
    double h;
    double theta;
    double nx;
    double ny;
    double r;
    double cx;
    double cy;
    int grain;
} RandomSurfaceState;

static double periodic_delta(double value, double length) {
    return value - length * round(value / length);
}

static void random_nearest_surface(
    const KernelParams *p,
    double lx,
    double ly,
    int n_grains,
    const double *grain_x,
    const double *grain_y,
    const double *grain_r,
    double x,
    double y,
    RandomSurfaceState *s
) {
    double xm = positive_mod(x, lx);
    double ym = positive_mod(y, ly);
    double best_h = INFINITY;
    double best_nx = 1.0;
    double best_ny = 0.0;
    double best_r = 1.0;
    double best_cx = NAN;
    double best_cy = NAN;
    int best_grain = -1;
    for (int g = 0; g < n_grains; ++g) {
        double dx = periodic_delta(xm - grain_x[g], lx);
        double dy = periodic_delta(ym - grain_y[g], ly);
        double rr = sqrt(dx * dx + dy * dy);
        if (rr < 1.0e-30) {
            rr = 1.0e-30;
        }
        double h = rr - (grain_r[g] + p->particle_radius);
        if (h < best_h) {
            best_h = h;
            best_nx = dx / rr;
            best_ny = dy / rr;
            best_r = rr;
            best_cx = x - dx;
            best_cy = y - dy;
            best_grain = g;
        }
    }
    s->h = best_h;
    s->theta = positive_mod(atan2(best_ny, best_nx), 2.0 * M_PI);
    s->nx = best_nx;
    s->ny = best_ny;
    s->r = best_r;
    s->cx = best_cx;
    s->cy = best_cy;
    s->grain = best_grain;
}

static void random_shell_entry_state(
    const KernelParams *p,
    double lx,
    double ly,
    int n_grains,
    const double *grain_x,
    const double *grain_y,
    const double *grain_r,
    double old_x,
    double old_y,
    double new_x,
    double new_y,
    const RandomSurfaceState *s0,
    const RandomSurfaceState *s1,
    RandomSurfaceState *entry,
    double *fraction
) {
    if (s0->h <= p->near_surface) {
        *entry = *s0;
        *fraction = 0.0;
        return;
    }
    if (s1->h > p->near_surface) {
        *entry = *s1;
        *fraction = 1.0;
        return;
    }
    double dx = new_x - old_x;
    double dy = periodic_delta(new_y - old_y, ly);
    double lo = 0.0;
    double hi = 1.0;
    RandomSurfaceState candidate = *s1;
    for (int iteration = 0; iteration < 28; ++iteration) {
        double mid = 0.5 * (lo + hi);
        random_nearest_surface(
            p,
            lx,
            ly,
            n_grains,
            grain_x,
            grain_y,
            grain_r,
            old_x + mid * dx,
            old_y + mid * dy,
            &candidate
        );
        if (candidate.h > p->near_surface) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    *fraction = hi;
    random_nearest_surface(
        p,
        lx,
        ly,
        n_grains,
        grain_x,
        grain_y,
        grain_r,
        old_x + hi * dx,
        old_y + hi * dy,
        entry
    );
}

static void random_interpolate_velocity(
    double lx,
    double ly,
    int nx,
    int ny,
    const double *ux_grid,
    const double *uy_grid,
    double x,
    double y,
    double *out_ux,
    double *out_uy
) {
    double gx = positive_mod(x, lx) / lx * (double)nx - 0.5;
    double gy = positive_mod(y, ly) / ly * (double)ny - 0.5;
    int i0_raw = (int)floor(gx);
    int j0_raw = (int)floor(gy);
    double tx = gx - (double)i0_raw;
    double ty = gy - (double)j0_raw;
    int i0 = wrap_index(i0_raw, nx);
    int j0 = wrap_index(j0_raw, ny);
    int i1 = wrap_index(i0 + 1, nx);
    int j1 = wrap_index(j0 + 1, ny);
    size_t i00 = (size_t)i0 * (size_t)ny + (size_t)j0;
    size_t i10 = (size_t)i1 * (size_t)ny + (size_t)j0;
    size_t i01 = (size_t)i0 * (size_t)ny + (size_t)j1;
    size_t i11 = (size_t)i1 * (size_t)ny + (size_t)j1;
    double w00 = (1.0 - tx) * (1.0 - ty);
    double w10 = tx * (1.0 - ty);
    double w01 = (1.0 - tx) * ty;
    double w11 = tx * ty;
    *out_ux = w00 * ux_grid[i00] + w10 * ux_grid[i10] + w01 * ux_grid[i01] + w11 * ux_grid[i11];
    *out_uy = w00 * uy_grid[i00] + w10 * uy_grid[i10] + w01 * uy_grid[i01] + w11 * uy_grid[i11];
}

static void random_project_to_surface(
    const KernelParams *p,
    double lx,
    double ly,
    int n_grains,
    const double *grain_x,
    const double *grain_y,
    const double *grain_r,
    double *x,
    double *y
) {
    RandomSurfaceState s;
    random_nearest_surface(p, lx, ly, n_grains, grain_x, grain_y, grain_r, *x, *y, &s);
    if (s.grain < 0) {
        return;
    }
    double old_xm = positive_mod(*x, lx);
    double old_ym = positive_mod(*y, ly);
    double target = grain_r[s.grain] + p->particle_radius + p->contact_gap;
    double projected_x = positive_mod(grain_x[s.grain] + s.nx * target, lx);
    double projected_y = positive_mod(grain_y[s.grain] + s.ny * target, ly);
    *x += periodic_delta(projected_x - old_xm, lx);
    *y = projected_y;
}

static int random_segment_contact(
    const KernelParams *p,
    double lx,
    double ly,
    int n_grains,
    const double *grain_x,
    const double *grain_y,
    const double *grain_r,
    double x0,
    double y0,
    double x1,
    double y1,
    double *hit_t,
    int *hit_grain,
    double *hit_nx,
    double *hit_ny
) {
    double p0x = positive_mod(x0, lx);
    double p0y = positive_mod(y0, ly);
    double dpx = periodic_delta(positive_mod(x1, lx) - p0x, lx);
    double dpy = periodic_delta(positive_mod(y1, ly) - p0y, ly);
    double aa = dpx * dpx + dpy * dpy;
    if (aa <= 1.0e-32) {
        return 0;
    }
    double best_t = INFINITY;
    int best_g = -1;
    double best_nx = 1.0;
    double best_ny = 0.0;
    for (int g = 0; g < n_grains; ++g) {
        double cx_img = p0x - periodic_delta(p0x - grain_x[g], lx);
        double cy_img = p0y - periodic_delta(p0y - grain_y[g], ly);
        double fx = p0x - cx_img;
        double fy = p0y - cy_img;
        double boundary_radius = grain_r[g] + p->particle_radius + p->contact_gap;
        double cc = fx * fx + fy * fy - boundary_radius * boundary_radius;
        double bb = fx * dpx + fy * dpy;
        double disc = bb * bb - aa * cc;
        if (disc < 0.0) {
            continue;
        }
        double t = (-bb - sqrt(disc)) / aa;
        int valid = (t > 1.0e-12 && t <= 1.0) || (cc <= 0.0 && bb < 0.0);
        if (cc <= 0.0 && bb < 0.0) {
            t = 0.0;
        }
        if (!valid || t >= best_t) {
            continue;
        }
        double contact_x = p0x + t * dpx;
        double contact_y = p0y + t * dpy;
        double nxh = contact_x - cx_img;
        double nyh = contact_y - cy_img;
        double nr = sqrt(nxh * nxh + nyh * nyh);
        if (nr < 1.0e-30) {
            nr = 1.0e-30;
        }
        best_t = t;
        best_g = g;
        best_nx = nxh / nr;
        best_ny = nyh / nr;
    }
    if (best_g < 0) {
        return 0;
    }
    *hit_t = best_t;
    *hit_grain = best_g;
    *hit_nx = best_nx;
    *hit_ny = best_ny;
    return 1;
}

static int random_reflect_segment(
    const KernelParams *p,
    double lx,
    double ly,
    int n_grains,
    const double *grain_x,
    const double *grain_y,
    const double *grain_r,
    double x0,
    double y0,
    double *x1,
    double *y1
) {
    double hit_t = 0.0;
    int hit_grain = -1;
    double hit_nx = 1.0;
    double hit_ny = 0.0;
    int hit = random_segment_contact(
        p,
        lx,
        ly,
        n_grains,
        grain_x,
        grain_y,
        grain_r,
        x0,
        y0,
        *x1,
        *y1,
        &hit_t,
        &hit_grain,
        &hit_nx,
        &hit_ny
    );
    if (!hit) {
        return 0;
    }
    (void)hit_grain;
    double p0x = positive_mod(x0, lx);
    double p0y = positive_mod(y0, ly);
    double dpx = periodic_delta(positive_mod(*x1, lx) - p0x, lx);
    double dpy = periodic_delta(positive_mod(*y1, ly) - p0y, ly);
    double contact_x = p0x + hit_t * dpx;
    double contact_y = p0y + hit_t * dpy;
    double rx = (1.0 - hit_t) * dpx;
    double ry = (1.0 - hit_t) * dpy;
    double inward = rx * hit_nx + ry * hit_ny;
    if (inward < 0.0) {
        rx -= 2.0 * inward * hit_nx;
        ry -= 2.0 * inward * hit_ny;
    }
    double new_xm = positive_mod(contact_x + rx + 1.0e-12 * hit_nx, lx);
    double new_ym = positive_mod(contact_y + ry + 1.0e-12 * hit_ny, ly);
    *x1 = x0 + periodic_delta(new_xm - p0x, lx);
    *y1 = new_ym;
    random_project_to_surface(p, lx, ly, n_grains, grain_x, grain_y, grain_r, x1, y1);
    return 1;
}

static NearWallStepChoice random_near_wall_choice(
    const KernelParams *p,
    double lx,
    double ly,
    int grid_nx,
    int grid_ny,
    const double *ux_grid,
    const double *uy_grid,
    int n_grains,
    const double *grain_x,
    const double *grain_y,
    const double *grain_r,
    int condition,
    const ParticleState *particle,
    double remaining_dt
) {
    NearWallStepChoice choice;
    choice.sub_dt = remaining_dt;
    choice.normal_mobility = 1.0;
    choice.normal_speed = 0.0;
    choice.target = fmax(p->resolved_langevin_normal_step, p->min_gap * 0.05);
    choice.min_dt_limited = 0;
    if (!particle->active) {
        return choice;
    }
    RandomSurfaceState s;
    random_nearest_surface(p, lx, ly, n_grains, grain_x, grain_y, grain_r, particle->x, particle->y, &s);
    double ux = 0.0;
    double uy = 0.0;
    random_interpolate_velocity(lx, ly, grid_nx, grid_ny, ux_grid, uy_grid, particle->x, particle->y, &ux, &uy);
    double normal_mobility = 1.0;
    double parallel_mobility = 1.0;
    double dnormal_dh = 0.0;
    near_wall_mobility(p, s.h, &normal_mobility, &parallel_mobility, &dnormal_dh);
    (void)parallel_mobility;
    double hydro_normal = ux * s.nx + uy * s.ny;
    double force_normal = dlvo_force_normal(p, s.h, condition);
    double stokes_mobility = 1.0 / (6.0 * M_PI * p->viscosity * p->particle_radius);
    double v_dlvo = stokes_mobility * force_normal;
    if (v_dlvo > p->dlvo_velocity_cap) {
        v_dlvo = p->dlvo_velocity_cap;
    }
    if (v_dlvo < -p->dlvo_velocity_cap) {
        v_dlvo = -p->dlvo_velocity_cap;
    }
    double normal_speed = normal_mobility * (hydro_normal + v_dlvo);
    if (p->wall_mobility_thermal_drift) {
        normal_speed += p->diffusivity * dnormal_dh;
    }
    double target = fmax(p->resolved_langevin_normal_step, p->min_gap * 0.05);
    double inward_speed = fmax(-normal_speed, 0.0);
    double dt_det = remaining_dt;
    if (inward_speed > 1.0e-30) {
        dt_det = 0.25 * target / inward_speed;
    }
    double dt_diff = remaining_dt;
    if (p->diffusivity > 0.0 && normal_mobility > 0.0) {
        dt_diff = 0.25 * target * target / (2.0 * p->diffusivity * normal_mobility);
    }
    int substeps = p->resolved_langevin_substeps > 0 ? p->resolved_langevin_substeps : 1;
    double base_dt = p->dt / (double)substeps;
    double out = remaining_dt;
    if (base_dt < out) {
        out = base_dt;
    }
    if (dt_det < out) {
        out = dt_det;
    }
    if (dt_diff < out) {
        out = dt_diff;
    }
    if (out < p->resolved_langevin_min_dt) {
        out = p->resolved_langevin_min_dt;
        choice.min_dt_limited = 1;
    }
    if (out > remaining_dt) {
        out = remaining_dt;
    }
    choice.sub_dt = out;
    choice.normal_mobility = normal_mobility;
    choice.normal_speed = normal_speed;
    choice.target = target;
    return choice;
}

static void random_advance_step(
    const KernelParams *p,
    double lx,
    double ly,
    int grid_nx,
    int grid_ny,
    const double *ux_grid,
    const double *uy_grid,
    int n_grains,
    const double *grain_x,
    const double *grain_y,
    const double *grain_r,
    int condition,
    int allow_attachment,
    int stop_at_next_distinct,
    double step_dt,
    uint64_t *rng_state,
    ParticleState *particle
) {
    if (!particle->active || step_dt <= 0.0) {
        return;
    }
    double old_x = particle->x;
    double old_y = particle->y;
    RandomSurfaceState s0;
    random_nearest_surface(p, lx, ly, n_grains, grain_x, grain_y, grain_r, old_x, old_y, &s0);

    double ux = 0.0;
    double uy = 0.0;
    random_interpolate_velocity(lx, ly, grid_nx, grid_ny, ux_grid, uy_grid, old_x, old_y, &ux, &uy);

    double normal_mobility = 1.0;
    double parallel_mobility = 1.0;
    double dnormal_dh = 0.0;
    near_wall_mobility(p, s0.h, &normal_mobility, &parallel_mobility, &dnormal_dh);
    double tx = -s0.ny;
    double ty = s0.nx;
    double hydro_n = ux * s0.nx + uy * s0.ny;
    double hydro_t = ux * tx + uy * ty;
    double ux_eff = normal_mobility * hydro_n * s0.nx + parallel_mobility * hydro_t * tx;
    double uy_eff = normal_mobility * hydro_n * s0.ny + parallel_mobility * hydro_t * ty;

    double force_normal = dlvo_force_normal(p, s0.h, condition);
    double stokes_mobility = 1.0 / (6.0 * M_PI * p->viscosity * p->particle_radius);
    double v_dlvo = stokes_mobility * force_normal;
    if (v_dlvo > p->dlvo_velocity_cap) {
        v_dlvo = p->dlvo_velocity_cap;
    }
    if (v_dlvo < -p->dlvo_velocity_cap) {
        v_dlvo = -p->dlvo_velocity_cap;
    }
    ux_eff += normal_mobility * v_dlvo * s0.nx;
    uy_eff += normal_mobility * v_dlvo * s0.ny;
    if (p->wall_mobility_thermal_drift) {
        double thermal_drift = p->diffusivity * dnormal_dh;
        ux_eff += thermal_drift * s0.nx;
        uy_eff += thermal_drift * s0.ny;
    }

    double noise0 = 0.0;
    double noise1 = 0.0;
    rng_normal_pair(rng_state, &noise0, &noise1);
    double brownian_normal = sqrt(2.0 * p->diffusivity * normal_mobility * step_dt) * noise0;
    double brownian_parallel = sqrt(2.0 * p->diffusivity * parallel_mobility * step_dt) * noise1;
    double new_x = old_x + ux_eff * step_dt + brownian_normal * s0.nx + brownian_parallel * tx;
    double new_y = positive_mod(old_y + uy_eff * step_dt + brownian_normal * s0.ny + brownian_parallel * ty, ly);

    double hit_t = 0.0;
    int hit_grain = -1;
    double hit_nx = 1.0;
    double hit_ny = 0.0;
    int segment_hit = random_segment_contact(
        p,
        lx,
        ly,
        n_grains,
        grain_x,
        grain_y,
        grain_r,
        old_x,
        old_y,
        new_x,
        new_y,
        &hit_t,
        &hit_grain,
        &hit_nx,
        &hit_ny
    );
    if (segment_hit) {
        particle->contact_events += 1;
    }

    RandomSurfaceState s1;
    random_nearest_surface(p, lx, ly, n_grains, grain_x, grain_y, grain_r, new_x, new_y, &s1);
    if (condition == 2 && s1.h < s0.h && s1.h <= p->near_surface) {
        double old_u = dlvo_potential_energy(p, s0.h, condition);
        double new_u = dlvo_potential_energy(p, s1.h, condition);
        if (new_u > old_u) {
            double kbt = 1.380649e-23 * p->temperature;
            double exponent = (new_u - old_u) / kbt;
            if (exponent > 700.0) {
                exponent = 700.0;
            }
            double accept_probability = exp(-exponent);
            if (rng_uniform(rng_state) > accept_probability) {
                new_x = old_x;
                new_y = old_y;
                random_nearest_surface(p, lx, ly, n_grains, grain_x, grain_y, grain_r, new_x, new_y, &s1);
                segment_hit = 0;
            }
        }
    }

    int attach = allow_attachment && (segment_hit || s1.h <= 2.0e-9);
    if (attach) {
        particle->attached = 1;
        particle->active = 0;
    } else {
        if (segment_hit) {
            random_reflect_segment(p, lx, ly, n_grains, grain_x, grain_y, grain_r, old_x, old_y, &new_x, &new_y);
            random_nearest_surface(p, lx, ly, n_grains, grain_x, grain_y, grain_r, new_x, new_y, &s1);
        }
        if (s1.h < p->contact_gap) {
            random_project_to_surface(p, lx, ly, n_grains, grain_x, grain_y, grain_r, &new_x, &new_y);
            random_nearest_surface(p, lx, ly, n_grains, grain_x, grain_y, grain_r, new_x, new_y, &s1);
        }
    }

    particle->x = new_x;
    particle->y = positive_mod(new_y, ly);
    particle->t += step_dt;

    if (s1.h < particle->h_min) {
        particle->h_min = s1.h;
    }
    int now_near = s1.h <= p->near_surface;
    int entering = now_near && !particle->in_event;
    int leaving = !now_near && particle->in_event;
    if (leaving) {
        particle->theta_exit = particle->theta_last_near;
        particle->collector_exit = particle->collector_last_near;
        if (
            particle->next_entry_collector < 0 &&
            particle->collector_last_near == particle->first_entry_collector
        ) {
            particle->departure_theta_to_next = particle->theta_last_near;
            particle->departure_time_to_next = particle->t;
        }
    }
    int completed_entry_pair = 0;
    if (entering) {
        RandomSurfaceState entry_state;
        double entry_fraction = 1.0;
        random_shell_entry_state(
            p,
            lx,
            ly,
            n_grains,
            grain_x,
            grain_y,
            grain_r,
            old_x,
            old_y,
            new_x,
            new_y,
            &s0,
            &s1,
            &entry_state,
            &entry_fraction
        );
        double entry_dx = new_x - old_x;
        double entry_dy = periodic_delta(new_y - old_y, ly);
        double entry_x = old_x + entry_fraction * entry_dx;
        double entry_y = positive_mod(old_y + entry_fraction * entry_dy, ly);
        double entry_time = particle->t - step_dt + entry_fraction * step_dt;
        particle->interceptions += 1;
        particle->theta_entry = entry_state.theta;
        particle->collector_entry = entry_state.grain;
        completed_entry_pair = record_entry_pair(
            particle,
            entry_state.theta,
            entry_state.grain,
            entry_x,
            entry_y,
            entry_time,
            entry_state.cx,
            entry_state.cy,
            2
        );
    }
    if (now_near) {
        particle->theta_last_near = s1.theta;
        particle->collector_last_near = s1.grain;
        particle->near_time += step_dt;
        if (
            particle->next_entry_collector < 0 &&
            s1.grain == particle->first_entry_collector
        ) {
            particle->first_collector_near_time += step_dt;
        }
    }
    particle->in_event = now_near;
    particle->theta_final = s1.theta;
    particle->collector_final = s1.grain;

    if (completed_entry_pair && stop_at_next_distinct) {
        particle->active = 0;
        particle->travel_time = particle->t;
        particle->dy = particle->y - particle->y0;
    }

    if (particle->active && !stop_at_next_distinct && particle->x >= lx) {
        particle->exited = 1;
        particle->active = 0;
        particle->y_out = positive_mod(particle->y, ly);
        particle->travel_time = particle->t;
        particle->dy = particle->y - particle->y0;
    }
}

static void random_mark_occupancy(
    const KernelParams *p,
    double lx,
    double ly,
    int n_grains,
    const double *grain_x,
    const double *grain_y,
    const double *grain_r,
    const ParticleState *particle,
    int pore_nx,
    int pore_ny,
    int surface_bins,
    uint8_t *pore_seen,
    uint8_t *surface_seen
) {
    if (pore_seen != 0 && pore_nx > 0 && pore_ny > 0) {
        double xm = positive_mod(particle->x, lx);
        double ym = positive_mod(particle->y, ly);
        int ix = (int)floor(xm / lx * (double)pore_nx);
        int iy = (int)floor(ym / ly * (double)pore_ny);
        if (ix < 0) ix = 0;
        if (ix >= pore_nx) ix = pore_nx - 1;
        if (iy < 0) iy = 0;
        if (iy >= pore_ny) iy = pore_ny - 1;
        pore_seen[(size_t)ix * (size_t)pore_ny + (size_t)iy] = 1;
    }
    if (surface_seen != 0 && surface_bins > 0 && n_grains > 0) {
        RandomSurfaceState s;
        random_nearest_surface(p, lx, ly, n_grains, grain_x, grain_y, grain_r, particle->x, particle->y, &s);
        if (s.h <= p->near_surface && s.grain >= 0 && s.grain < n_grains) {
            int sector = (int)floor(s.theta / (2.0 * M_PI) * (double)surface_bins);
            if (sector < 0) sector = 0;
            if (sector >= surface_bins) sector = surface_bins - 1;
            surface_seen[(size_t)s.grain * (size_t)surface_bins + (size_t)sector] = 1;
        }
    }
}

int simulate_random_particle_kernel(
    const KernelParams *p,
    double lx,
    double ly,
    int grid_nx,
    int grid_ny,
    const double *ux_grid,
    const double *uy_grid,
    int n_grains,
    const double *grain_x,
    const double *grain_y,
    const double *grain_r,
    int condition,
    int allow_attachment,
    int stop_at_next_distinct,
    int n_particles,
    uint64_t seed,
    const double *initial_x,
    const double *initial_y,
    double *y_in,
    double *y_out,
    double *travel_time,
    double *dy,
    uint8_t *attached,
    uint8_t *exited,
    uint8_t *censored,
    int32_t *interceptions,
    double *near_time,
    double *h_min,
    double *theta_entry,
    double *theta_exit,
    double *theta_final,
    int32_t *grain_entry,
    int32_t *grain_exit,
    int32_t *grain_final,
    int32_t *contact_events,
    double *x_final,
    double *y_final,
    int32_t *near_wall_outer_steps,
    int32_t *adaptive_substeps,
    int32_t *guard_hits,
    int32_t *min_dt_hits,
    double *max_inward_det_normal_step,
    double *max_brownian_normal_std,
    double *first_entry_theta,
    double *first_entry_time,
    double *first_entry_x,
    double *first_entry_y,
    double *first_entry_collector_cx,
    double *first_entry_collector_cy,
    int32_t *first_entry_collector,
    double *next_entry_theta,
    double *next_entry_time,
    double *next_entry_x,
    double *next_entry_y,
    double *next_entry_collector_cx,
    double *next_entry_collector_cy,
    int32_t *next_entry_collector,
    double *departure_theta_to_next,
    double *departure_time_to_next,
    double *first_collector_near_time,
    int32_t *same_collector_reentries_before_next
) {
    if (
        p == 0 || ux_grid == 0 || uy_grid == 0 || grain_x == 0 || grain_y == 0 || grain_r == 0 ||
        initial_x == 0 || initial_y == 0
    ) {
        return 1;
    }
    if (lx <= 0.0 || ly <= 0.0 || grid_nx <= 1 || grid_ny <= 1 || n_grains <= 0 || n_particles < 0) {
        return 2;
    }
    if (p->dt <= 0.0 || p->max_time <= 0.0) {
        return 3;
    }

    int max_steps = (int)ceil(p->max_time / p->dt);
    if (max_steps < 1) {
        max_steps = 1;
    }
    for (int particle_id = 0; particle_id < n_particles; ++particle_id) {
        ParticleState particle;
        particle.x = initial_x[particle_id];
        particle.y = positive_mod(initial_y[particle_id], ly);
        particle.y0 = positive_mod(initial_y[particle_id], ly);
        particle.t = 0.0;
        particle.active = 1;
        particle.attached = 0;
        particle.exited = 0;
        particle.censored = 0;
        particle.interceptions = 0;
        particle.in_event = 0;
        particle.well_interceptions = 0;
        particle.in_well_event = 0;
        particle.center_interceptions = 0;
        particle.corner_interceptions = 0;
        particle.center_well_interceptions = 0;
        particle.corner_well_interceptions = 0;
        particle.contact_events = 0;
        particle.near_time = 0.0;
        particle.well_time = 0.0;
        particle.well_angular_travel = 0.0;
        particle.well_net_angular_travel = 0.0;
        particle.h_min = INFINITY;
        particle.theta_entry = NAN;
        particle.theta_exit = NAN;
        particle.theta_last_near = NAN;
        particle.theta_final = NAN;
        particle.theta_well_entry = NAN;
        particle.theta_well_exit = NAN;
        particle.theta_well_last = NAN;
        particle.theta_well_final = NAN;
        particle.collector_entry = -1;
        particle.collector_exit = -1;
        particle.collector_last_near = -1;
        particle.collector_final = -1;
        particle.collector_well_entry = -1;
        particle.collector_well_exit = -1;
        particle.collector_well_last = -1;
        particle.collector_well_final = -1;
        particle.y_out = NAN;
        particle.travel_time = NAN;
        particle.dy = NAN;
        particle.near_wall_outer_steps = 0;
        particle.adaptive_substeps = 0;
        particle.guard_hits = 0;
        particle.min_dt_hits = 0;
        particle.sub_dt_sum = 0.0;
        particle.sub_dt_sq_sum = 0.0;
        particle.min_sub_dt = INFINITY;
        particle.max_sub_dt = 0.0;
        particle.max_inward_det_normal_step = 0.0;
        particle.max_brownian_normal_std = 0.0;
        particle.first_entry_theta = NAN;
        particle.first_entry_time = NAN;
        particle.first_entry_x = NAN;
        particle.first_entry_y = NAN;
        particle.first_entry_collector_cx = NAN;
        particle.first_entry_collector_cy = NAN;
        particle.first_entry_collector = -1;
        particle.next_entry_theta = NAN;
        particle.next_entry_time = NAN;
        particle.next_entry_x = NAN;
        particle.next_entry_y = NAN;
        particle.next_entry_collector_cx = NAN;
        particle.next_entry_collector_cy = NAN;
        particle.next_entry_collector = -1;
        particle.departure_theta_to_next = NAN;
        particle.departure_time_to_next = NAN;
        particle.first_collector_near_time = 0.0;
        particle.same_collector_reentries_before_next = 0;

        uint64_t rng_state = seed ^ (UINT64_C(0xD1B54A32D192ED03) * (uint64_t)(particle_id + 1));
        rng_state += UINT64_C(0x9E3779B97F4A7C15);

        for (int step = 0; step < max_steps && particle.active; ++step) {
            RandomSurfaceState current;
            random_nearest_surface(
                p,
                lx,
                ly,
                n_grains,
                grain_x,
                grain_y,
                grain_r,
                particle.x,
                particle.y,
                &current
            );
            if (current.h <= p->resolved_langevin_substep_cutoff) {
                particle.near_wall_outer_steps += 1;
                double remaining_dt = p->dt;
                int guard_max = p->resolved_langevin_max_substeps > 0 ? p->resolved_langevin_max_substeps : 1;
                int guard = 0;
                while (remaining_dt > 1.0e-15 && particle.active && guard < guard_max) {
                    NearWallStepChoice choice = random_near_wall_choice(
                        p,
                        lx,
                        ly,
                        grid_nx,
                        grid_ny,
                        ux_grid,
                        uy_grid,
                        n_grains,
                        grain_x,
                        grain_y,
                        grain_r,
                        condition,
                        &particle,
                        remaining_dt
                    );
                    double sub_dt = choice.sub_dt;
                    if (sub_dt > remaining_dt) {
                        sub_dt = remaining_dt;
                    }
                    particle.adaptive_substeps += 1;
                    particle.min_dt_hits += choice.min_dt_limited;
                    double det_normal_step = fmax(-choice.normal_speed, 0.0) * sub_dt;
                    if (det_normal_step > particle.max_inward_det_normal_step) {
                        particle.max_inward_det_normal_step = det_normal_step;
                    }
                    double brownian_std = 0.0;
                    if (p->diffusivity > 0.0 && choice.normal_mobility > 0.0) {
                        brownian_std = sqrt(2.0 * p->diffusivity * choice.normal_mobility * sub_dt);
                    }
                    if (brownian_std > particle.max_brownian_normal_std) {
                        particle.max_brownian_normal_std = brownian_std;
                    }
                    random_advance_step(
                        p,
                        lx,
                        ly,
                        grid_nx,
                        grid_ny,
                        ux_grid,
                        uy_grid,
                        n_grains,
                        grain_x,
                        grain_y,
                        grain_r,
                        condition,
                        allow_attachment,
                        stop_at_next_distinct,
                        sub_dt,
                        &rng_state,
                        &particle
                    );
                    remaining_dt -= sub_dt;
                    ++guard;
                }
                if (remaining_dt > 1.0e-15 && particle.active) {
                    particle.guard_hits += 1;
                    random_advance_step(
                        p,
                        lx,
                        ly,
                        grid_nx,
                        grid_ny,
                        ux_grid,
                        uy_grid,
                        n_grains,
                        grain_x,
                        grain_y,
                        grain_r,
                        condition,
                        allow_attachment,
                        stop_at_next_distinct,
                        remaining_dt,
                        &rng_state,
                        &particle
                    );
                }
            } else {
                random_advance_step(
                    p,
                    lx,
                    ly,
                    grid_nx,
                    grid_ny,
                    ux_grid,
                    uy_grid,
                    n_grains,
                    grain_x,
                    grain_y,
                    grain_r,
                    condition,
                    allow_attachment,
                    stop_at_next_distinct,
                    p->dt,
                    &rng_state,
                    &particle
                );
            }
        }

        if (particle.active) {
            particle.censored = 1;
            particle.travel_time = p->max_time;
        }
        RandomSurfaceState final_state;
        random_nearest_surface(
            p,
            lx,
            ly,
            n_grains,
            grain_x,
            grain_y,
            grain_r,
            particle.x,
            particle.y,
            &final_state
        );
        particle.theta_final = final_state.theta;
        particle.collector_final = final_state.grain;
        if (final_state.h < particle.h_min) {
            particle.h_min = final_state.h;
        }

        y_in[particle_id] = particle.y0;
        y_out[particle_id] = particle.y_out;
        travel_time[particle_id] = particle.travel_time;
        dy[particle_id] = particle.dy;
        attached[particle_id] = (uint8_t)particle.attached;
        exited[particle_id] = (uint8_t)particle.exited;
        censored[particle_id] = (uint8_t)particle.censored;
        interceptions[particle_id] = (int32_t)particle.interceptions;
        near_time[particle_id] = particle.near_time;
        h_min[particle_id] = particle.h_min;
        theta_entry[particle_id] = particle.theta_entry;
        theta_exit[particle_id] = particle.theta_exit;
        theta_final[particle_id] = particle.theta_final;
        grain_entry[particle_id] = (int32_t)particle.collector_entry;
        grain_exit[particle_id] = (int32_t)particle.collector_exit;
        grain_final[particle_id] = (int32_t)particle.collector_final;
        contact_events[particle_id] = (int32_t)particle.contact_events;
        x_final[particle_id] = positive_mod(particle.x, lx);
        y_final[particle_id] = positive_mod(particle.y, ly);
        near_wall_outer_steps[particle_id] = (int32_t)particle.near_wall_outer_steps;
        adaptive_substeps[particle_id] = (int32_t)particle.adaptive_substeps;
        guard_hits[particle_id] = (int32_t)particle.guard_hits;
        min_dt_hits[particle_id] = (int32_t)particle.min_dt_hits;
        max_inward_det_normal_step[particle_id] = particle.max_inward_det_normal_step;
        max_brownian_normal_std[particle_id] = particle.max_brownian_normal_std;
        first_entry_theta[particle_id] = particle.first_entry_theta;
        first_entry_time[particle_id] = particle.first_entry_time;
        first_entry_x[particle_id] = particle.first_entry_x;
        first_entry_y[particle_id] = particle.first_entry_y;
        first_entry_collector_cx[particle_id] = particle.first_entry_collector_cx;
        first_entry_collector_cy[particle_id] = particle.first_entry_collector_cy;
        first_entry_collector[particle_id] = (int32_t)particle.first_entry_collector;
        next_entry_theta[particle_id] = particle.next_entry_theta;
        next_entry_time[particle_id] = particle.next_entry_time;
        next_entry_x[particle_id] = particle.next_entry_x;
        next_entry_y[particle_id] = particle.next_entry_y;
        next_entry_collector_cx[particle_id] = particle.next_entry_collector_cx;
        next_entry_collector_cy[particle_id] = particle.next_entry_collector_cy;
        next_entry_collector[particle_id] = (int32_t)particle.next_entry_collector;
        departure_theta_to_next[particle_id] = particle.departure_theta_to_next;
        departure_time_to_next[particle_id] = particle.departure_time_to_next;
        first_collector_near_time[particle_id] = particle.first_collector_near_time;
        same_collector_reentries_before_next[particle_id] =
            (int32_t)particle.same_collector_reentries_before_next;
    }
    return 0;
}

int simulate_random_occupancy_kernel(
    const KernelParams *p,
    double lx,
    double ly,
    int grid_nx,
    int grid_ny,
    const double *ux_grid,
    const double *uy_grid,
    int n_grains,
    const double *grain_x,
    const double *grain_y,
    const double *grain_r,
    int condition,
    int allow_attachment,
    int n_particles,
    uint64_t seed,
    const double *initial_x,
    const double *initial_y,
    int pore_nx,
    int pore_ny,
    int surface_bins,
    int sample_stride,
    double *pore_counts,
    double *surface_counts,
    uint8_t *attached,
    uint8_t *exited,
    uint8_t *censored,
    int32_t *interceptions,
    double *near_time,
    double *h_min,
    int32_t *contact_events
) {
    if (
        p == 0 || ux_grid == 0 || uy_grid == 0 || grain_x == 0 || grain_y == 0 || grain_r == 0 ||
        initial_x == 0 || initial_y == 0 || pore_counts == 0 || surface_counts == 0
    ) {
        return 1;
    }
    if (
        lx <= 0.0 || ly <= 0.0 || grid_nx <= 1 || grid_ny <= 1 || n_grains <= 0 ||
        n_particles < 0 || pore_nx <= 0 || pore_ny <= 0 || surface_bins <= 0
    ) {
        return 2;
    }
    if (p->dt <= 0.0 || p->max_time <= 0.0) {
        return 3;
    }
    if (sample_stride < 1) {
        sample_stride = 1;
    }
    int max_steps = (int)ceil(p->max_time / p->dt);
    if (max_steps < 1) {
        max_steps = 1;
    }
    size_t pore_size = (size_t)pore_nx * (size_t)pore_ny;
    size_t surface_size = (size_t)n_grains * (size_t)surface_bins;

    for (int particle_id = 0; particle_id < n_particles; ++particle_id) {
        uint8_t *pore_seen = (uint8_t *)calloc(pore_size, sizeof(uint8_t));
        uint8_t *surface_seen = (uint8_t *)calloc(surface_size, sizeof(uint8_t));
        if (pore_seen == 0 || surface_seen == 0) {
            free(pore_seen);
            free(surface_seen);
            return 4;
        }

        ParticleState particle;
        particle.x = initial_x[particle_id];
        particle.y = positive_mod(initial_y[particle_id], ly);
        particle.y0 = positive_mod(initial_y[particle_id], ly);
        particle.t = 0.0;
        particle.active = 1;
        particle.attached = 0;
        particle.exited = 0;
        particle.censored = 0;
        particle.interceptions = 0;
        particle.in_event = 0;
        particle.well_interceptions = 0;
        particle.in_well_event = 0;
        particle.center_interceptions = 0;
        particle.corner_interceptions = 0;
        particle.center_well_interceptions = 0;
        particle.corner_well_interceptions = 0;
        particle.contact_events = 0;
        particle.near_time = 0.0;
        particle.well_time = 0.0;
        particle.well_angular_travel = 0.0;
        particle.well_net_angular_travel = 0.0;
        particle.h_min = INFINITY;
        particle.theta_entry = NAN;
        particle.theta_exit = NAN;
        particle.theta_last_near = NAN;
        particle.theta_final = NAN;
        particle.theta_well_entry = NAN;
        particle.theta_well_exit = NAN;
        particle.theta_well_last = NAN;
        particle.theta_well_final = NAN;
        particle.collector_entry = -1;
        particle.collector_exit = -1;
        particle.collector_last_near = -1;
        particle.collector_final = -1;
        particle.collector_well_entry = -1;
        particle.collector_well_exit = -1;
        particle.collector_well_last = -1;
        particle.collector_well_final = -1;
        particle.y_out = NAN;
        particle.travel_time = NAN;
        particle.dy = NAN;
        particle.near_wall_outer_steps = 0;
        particle.adaptive_substeps = 0;
        particle.guard_hits = 0;
        particle.min_dt_hits = 0;
        particle.sub_dt_sum = 0.0;
        particle.sub_dt_sq_sum = 0.0;
        particle.min_sub_dt = INFINITY;
        particle.max_sub_dt = 0.0;
        particle.max_inward_det_normal_step = 0.0;
        particle.max_brownian_normal_std = 0.0;

        particle.first_entry_theta = NAN;
        particle.first_entry_time = NAN;
        particle.first_entry_x = NAN;
        particle.first_entry_y = NAN;
        particle.first_entry_collector_cx = NAN;
        particle.first_entry_collector_cy = NAN;
        particle.first_entry_collector = -1;
        particle.next_entry_theta = NAN;
        particle.next_entry_time = NAN;
        particle.next_entry_x = NAN;
        particle.next_entry_y = NAN;
        particle.next_entry_collector_cx = NAN;
        particle.next_entry_collector_cy = NAN;
        particle.next_entry_collector = -1;
        particle.departure_theta_to_next = NAN;
        particle.departure_time_to_next = NAN;
        particle.first_collector_near_time = 0.0;
        particle.same_collector_reentries_before_next = 0;

        uint64_t rng_state = seed ^ (UINT64_C(0xD1B54A32D192ED03) * (uint64_t)(particle_id + 1));
        rng_state += UINT64_C(0x9E3779B97F4A7C15);

        random_mark_occupancy(
            p, lx, ly, n_grains, grain_x, grain_y, grain_r, &particle,
            pore_nx, pore_ny, surface_bins, pore_seen, surface_seen
        );

        for (int step = 0; step < max_steps && particle.active; ++step) {
            RandomSurfaceState current;
            random_nearest_surface(
                p,
                lx,
                ly,
                n_grains,
                grain_x,
                grain_y,
                grain_r,
                particle.x,
                particle.y,
                &current
            );
            if (current.h <= p->resolved_langevin_substep_cutoff) {
                particle.near_wall_outer_steps += 1;
                double remaining_dt = p->dt;
                int guard_max = p->resolved_langevin_max_substeps > 0 ? p->resolved_langevin_max_substeps : 1;
                int guard = 0;
                while (remaining_dt > 1.0e-15 && particle.active && guard < guard_max) {
                    NearWallStepChoice choice = random_near_wall_choice(
                        p,
                        lx,
                        ly,
                        grid_nx,
                        grid_ny,
                        ux_grid,
                        uy_grid,
                        n_grains,
                        grain_x,
                        grain_y,
                        grain_r,
                        condition,
                        &particle,
                        remaining_dt
                    );
                    double sub_dt = choice.sub_dt;
                    if (sub_dt > remaining_dt) {
                        sub_dt = remaining_dt;
                    }
                    particle.adaptive_substeps += 1;
                    particle.min_dt_hits += choice.min_dt_limited;
                    random_advance_step(
                        p,
                        lx,
                        ly,
                        grid_nx,
                        grid_ny,
                        ux_grid,
                        uy_grid,
                        n_grains,
                        grain_x,
                        grain_y,
                        grain_r,
                        condition,
                        allow_attachment,
                        0,
                        sub_dt,
                        &rng_state,
                        &particle
                    );
                    remaining_dt -= sub_dt;
                    ++guard;
                }
                if (remaining_dt > 1.0e-15 && particle.active) {
                    particle.guard_hits += 1;
                    random_advance_step(
                        p,
                        lx,
                        ly,
                        grid_nx,
                        grid_ny,
                        ux_grid,
                        uy_grid,
                        n_grains,
                        grain_x,
                        grain_y,
                        grain_r,
                        condition,
                        allow_attachment,
                        0,
                        remaining_dt,
                        &rng_state,
                        &particle
                    );
                }
            } else {
                random_advance_step(
                    p,
                    lx,
                    ly,
                    grid_nx,
                    grid_ny,
                    ux_grid,
                    uy_grid,
                    n_grains,
                    grain_x,
                    grain_y,
                    grain_r,
                    condition,
                    allow_attachment,
                    0,
                    p->dt,
                    &rng_state,
                    &particle
                );
            }
            if ((step + 1) % sample_stride == 0 && particle.active) {
                random_mark_occupancy(
                    p, lx, ly, n_grains, grain_x, grain_y, grain_r, &particle,
                    pore_nx, pore_ny, surface_bins, pore_seen, surface_seen
                );
            }
        }

        if (particle.active) {
            particle.censored = 1;
            particle.travel_time = p->max_time;
        }
        random_mark_occupancy(
            p, lx, ly, n_grains, grain_x, grain_y, grain_r, &particle,
            pore_nx, pore_ny, surface_bins, pore_seen, surface_seen
        );

        RandomSurfaceState final_state;
        random_nearest_surface(
            p,
            lx,
            ly,
            n_grains,
            grain_x,
            grain_y,
            grain_r,
            particle.x,
            particle.y,
            &final_state
        );
        if (final_state.h < particle.h_min) {
            particle.h_min = final_state.h;
        }

        for (size_t i = 0; i < pore_size; ++i) {
            if (pore_seen[i]) {
                pore_counts[i] += 1.0;
            }
        }
        for (size_t i = 0; i < surface_size; ++i) {
            if (surface_seen[i]) {
                surface_counts[i] += 1.0;
            }
        }
        attached[particle_id] = (uint8_t)particle.attached;
        exited[particle_id] = (uint8_t)particle.exited;
        censored[particle_id] = (uint8_t)particle.censored;
        interceptions[particle_id] = (int32_t)particle.interceptions;
        near_time[particle_id] = particle.near_time;
        h_min[particle_id] = particle.h_min;
        contact_events[particle_id] = (int32_t)particle.contact_events;
        free(pore_seen);
        free(surface_seen);
    }
    return 0;
}
