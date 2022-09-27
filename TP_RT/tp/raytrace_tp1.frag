/**
* @param in D direction du rayon
* @param in O origine du rayon
* @return couleur calculée en RGB
*/
layout(location=21) uniform vec3 light_position; // position de la lumière
layout(location=22) uniform vec3 C;	// couleur de la lumière

vec3 computeLocalRendering(in vec3 Dir, in vec3 Orig)
{
	vec3 N;
	vec3 Pg;
	intersection_info(N,Pg);
	vec4 col = intersection_color_info();
	vec4 mat = intersection_mat_info();
	vec3 L = normalize(light_position - Pg);
	float lamb = max(dot(N,L),0.0);

	vec3 R = reflect(-L,N);
	vec3 E = normalize(Orig - Pg);
	float sp = max(dot(R,E),0.0);
	float expos = mix(500,5,mat.g);
	vec3 specul = vec3(pow(sp,expos))* (1.0 - mat.g/2.0);
	vec3 color = col.rgb*lamb + specul;
	return mix(vec3(0),color,float(hit()));
}
vec3 raytrace(in vec3 Dir, in vec3 Orig)   
{
	just_hit_bvh(Orig,Dir);
	traverse_all_bvh(Orig,Dir);
	vec3 N;
	vec3 Pg;
	intersection_info(N,Pg);
	vec4 col = intersection_color_info();

	vec3 L = normalize(light_position-Pg);
	float lamb = dot(L,N);
	vec3 color = lamb*col.rgb;

	return mix(vec3(0),color,float(hit()));
}

